// src/runtime_manager.cpp（Day5版）
#include "runtime_manager.h"
#include "scheduler.h"
#include "dispatcher.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <sched.h>
#include <pthread.h>

static int64_t get_time_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count();
}

RuntimeManager::RuntimeManager(int core)
    : core_(core), start_time_us_(get_time_us()) {}

RuntimeManager::~RuntimeManager() { stop(); }

int64_t RuntimeManager::now_us() const {
    return get_time_us() - start_time_us_;
}

void RuntimeManager::set_scheduler(std::shared_ptr<BaselineScheduler> s) {
    scheduler_ = s;
}
void RuntimeManager::set_dispatcher(std::shared_ptr<Dispatcher> d) {
    dispatcher_ = d;
}

void RuntimeManager::set_gpu_executor(std::shared_ptr<GpuExecutor> ge) {
    gpu_executor_ = ge;
}

void RuntimeManager::post_event(RuntimeEvent evt) {
    { std::lock_guard<std::mutex> lk(eq_mutex_);
      eq_.push(std::move(evt)); }
    eq_cv_.notify_one();
}

void RuntimeManager::start() {
    running_ = true;
    start_time_us_ = get_time_us();
    thread_ = std::thread([this]{ event_loop(); });
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_, &cs);
    pthread_setaffinity_np(thread_.native_handle(), sizeof(cs), &cs);
    std::cout << "[RTMgr] Started on core" << core_ << "\n";
}

void RuntimeManager::stop() {
    if (!running_.exchange(false)) return;
    post_event({EventType::SHUTDOWN});
    if (thread_.joinable()) thread_.join();
    std::cout << "[RTMgr] Stopped. released=" << jobs_released_
              << " completed=" << jobs_completed_ << "\n";
}

NodeInstance* RuntimeManager::find_node_instance(const std::string& job_id,
                                                   const std::string& node_id) {
    // 注意：调用方需自己持有 jobs_mutex_，此函数不加锁
    auto it = active_jobs_.find(job_id);
    if (it == active_jobs_.end()) return nullptr;
    auto nit = it->second->node_instances.find(node_id);
    if (nit == it->second->node_instances.end()) return nullptr;
    return &nit->second;
}

void RuntimeManager::enqueue_node(NodeInstance* ni) {
    // GPU 节点和 CPU 节点完全一样，都进 Scheduler 对应核的队列
    // GPU 节点的特殊处理（锁请求、kernel提交）在 CpuWorker 执行时触发
    scheduler_->enqueue(ni);
}

void RuntimeManager::event_loop() {
    while (true) {
        RuntimeEvent evt;
        { std::unique_lock<std::mutex> lk(eq_mutex_);
          eq_cv_.wait(lk, [this]{ return !eq_.empty(); });
          evt = std::move(eq_.front()); eq_.pop(); }

        if (evt.type == EventType::SHUTDOWN)       return;
        if (evt.type == EventType::JOB_RELEASE)    handle_job_release(evt);
        if (evt.type == EventType::NODE_COMPLETE)  handle_node_complete(evt);
        if (evt.type == EventType::RESOURCE_IDLE)  handle_resource_idle(evt);
        if (evt.type == EventType::GPU_LOCK_REQUEST)  handle_gpu_lock_request(evt);  // ← 新增
        if (evt.type == EventType::GPU_KERNEL_DONE)   handle_gpu_kernel_done(evt);   // ← 新增
    }
}

void RuntimeManager::handle_job_release(const RuntimeEvent& evt) {
    int64_t now = now_us();
    jobs_released_++;
    std::cout << "[RTMgr] JOB_RELEASE " << evt.job_id
              << " at +" << now << "us\n";

    auto job = std::make_unique<JobInstance>(
        evt.job_id, evt.task_tmpl, now, evt.task_tmpl->deadline_us);

    auto initial = get_initial_ready_nodes(*job);
    {
        std::lock_guard<std::mutex> lk(jobs_mutex_);
        active_jobs_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(evt.job_id),
            std::forward_as_tuple(std::move(job)));
    }

    for (auto* ni : initial) {
        ni->trace.t_ready_us = now;
        enqueue_node(ni);
    }

    try_dispatch();
}

void RuntimeManager::handle_node_complete(const RuntimeEvent& evt) {
    int64_t now = now_us();
    std::lock_guard<std::mutex> lk(jobs_mutex_);
    auto it = active_jobs_.find(evt.job_id);
    if (it == active_jobs_.end()) return;

    auto& job = *it->second;
    auto newly_ready = advance_dag(job, evt.node_id, now);

    for (auto* ni : newly_ready) {
        enqueue_node(ni);
    }

    if (job.is_complete()) {
        bool met = (now <= job.absolute_deadline_us);
        std::cout << "[RTMgr] JOB_COMPLETE " << evt.job_id
                  << " response=" << (now - job.release_time_us) << "us"
                  << " deadline_met=" << (met ? "YES" : "NO!") << "\n";
        jobs_completed_++;
        active_jobs_.erase(it);
    }
    // 不在这里调 try_dispatch
    // Worker 会紧接着发 RESOURCE_IDLE，届时再调用
}

void RuntimeManager::handle_resource_idle(const RuntimeEvent&) {
    try_dispatch();
}

void RuntimeManager::handle_gpu_lock_request(const RuntimeEvent& evt) {
    int64_t now = now_us();

    NodeInstance* ni = nullptr;
    {
        std::lock_guard<std::mutex> lk(jobs_mutex_);
        ni = find_node_instance(evt.job_id, evt.node_id);
    }
    if (!ni) return;

    // CPU 提交阶段已在 CpuWorker 侧完成，这里记录 t_pre_end
    ni->trace.t_pre_end_us = now;

    if (!gpu_lock_.exchange(true)) {
        // 取到锁：立即开始，无排队等待
        ni->trace.t_accel_start_us = now;
        ni->trace.queue_wait_us    = 0;
        ni->state.store(NodeState::ACCEL_EXEC);

        std::cout << "[RTMgr] GPU_LOCK granted immediately "
                  << ni->node_id << "/" << ni->job_id << "\n";

        gpu_executor_->submit_kernel(ni);

    } else {
        // 未取到锁：记录入队时刻，等待锁释放
        ni->trace.t_queue_enter_us = now;
        ni->state.store(NodeState::ACCEL_WAIT);

        std::cout << "[RTMgr] GPU_LOCK queued "
                  << ni->node_id << "/" << ni->job_id
                  << " priority=" << ni->trace.task_priority << "\n";

        std::lock_guard<std::mutex> lk(gpu_queue_mutex_);
        gpu_queue_.push({ni, ni->trace.task_priority, now});
    }
}

void RuntimeManager::handle_gpu_kernel_done(const RuntimeEvent& evt) {
    int64_t now = now_us();

    NodeInstance* ni = nullptr;
    {
        std::lock_guard<std::mutex> lk(jobs_mutex_);
        ni = find_node_instance(evt.job_id, evt.node_id);
    }
    if (!ni) return;

    ni->trace.t_accel_end_us = now;
    ni->trace.accel_exec_us  = now - ni->trace.t_accel_start_us;

    std::cout << "[RTMgr] GPU_KERNEL_DONE " << ni->node_id
              << "/" << ni->job_id
              << " accel=" << ni->trace.accel_exec_us << "us\n";

    // 节点进入 POST_EXEC 阶段，重新入 Scheduler 队列等 CpuWorker 处理
    ni->state.store(NodeState::POST_EXEC);
    ni->trace.t_post_start_us = now;
    scheduler_->enqueue(ni);

    // 释放 GPU 锁，检查等待队列
    {
        std::lock_guard<std::mutex> lk(gpu_queue_mutex_);
        if (!gpu_queue_.empty()) {
            // 按优先级把锁给队头（高优先级先得锁，与 CPU 核调度对称）
            GpuQueueEntry next = gpu_queue_.top();
            gpu_queue_.pop();

            NodeInstance* next_ni = next.node_inst;
            next_ni->trace.t_accel_start_us = now;
            next_ni->trace.queue_wait_us =
                now - next_ni->trace.t_queue_enter_us;
            next_ni->state.store(NodeState::ACCEL_EXEC);

            std::cout << "[RTMgr] GPU_LOCK → " << next_ni->node_id
                      << "/" << next_ni->job_id
                      << " queue_wait=" << next_ni->trace.queue_wait_us
                      << "us\n";

            gpu_executor_->submit_kernel(next_ni);
            // 锁保持 true，继续被占用

        } else {
            // 无等待者，释放锁
            gpu_lock_.store(false);
            std::cout << "[RTMgr] GPU_LOCK released\n";
        }
    }

    // CpuWorker 队列有新节点，触发 try_dispatch
    try_dispatch();
}

void RuntimeManager::try_dispatch() {
    if (!scheduler_ || !dispatcher_) return;

    // 遍历每个 CPU Worker：空闲则取对应核队列的队头提交
    for (auto& [core, worker] : dispatcher_->get_cpu_workers()) {
        if (!worker->is_idle()) continue;

        // resource_id 与 core 号一致（注册时 register_cpu_worker(core, w)）
        NodeInstance* ni = scheduler_->peek_top(core);
        if (!ni) continue;

        scheduler_->pop_top(core);
        worker->submit(ni);
        std::cout << "[RTMgr] try_dispatch: " << ni->node_id
                  << " → core" << core << "\n";
    }
    // GPU 队列由 GpuOwnerWorker 内部管理，不在此处理
}

// ============================================================
// ReleaseTimer
// ============================================================
ReleaseTimer::ReleaseTimer(RuntimeManager* mgr, const TaskTemplate* tmpl, int n)
    : mgr_(mgr), tmpl_(tmpl), num_periods_(n) {}

void ReleaseTimer::start() {
    running_ = true;
    thread_ = std::thread([this]{ run(); });
}

void ReleaseTimer::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void ReleaseTimer::run() {
    int count = 0;
    while (running_ && count < num_periods_) {
        std::ostringstream oss;
        oss << tmpl_->task_id << "_job" << (++counter_);
        mgr_->post_event({EventType::JOB_RELEASE, oss.str(), "",
                          mgr_->now_us(), tmpl_});
        std::this_thread::sleep_for(
            std::chrono::microseconds(tmpl_->period_us));
        count++;
    }
}