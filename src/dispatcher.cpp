// src/dispatcher.cpp
#include "dispatcher.h"
#include <iostream>
#include <sched.h>
#include <pthread.h>
#include <chrono>

static int64_t get_time_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count();
}

static void bind_to_core(int core) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// ============================================================
// CpuWorker
// ============================================================
CpuWorker::CpuWorker(int cpu_core, RuntimeManager* mgr)
    : cpu_core_(cpu_core), mgr_(mgr) {}

CpuWorker::~CpuWorker() { stop(); }

void CpuWorker::start() {
    running_ = true;
    thread_ = std::thread([this]{ worker_loop(); });
}

void CpuWorker::stop() {
    if (!running_.exchange(false)) return;
    q_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void CpuWorker::submit(NodeInstance* ni) {
    WorkerRequest req;
    req.node_inst       = ni;
    req.task_priority   = ni->trace.task_priority;
    req.enqueue_time_us = get_time_us();
    { std::lock_guard<std::mutex> lk(q_mutex_);
      queue_.push(std::move(req)); }
    q_cv_.notify_one();
}

void CpuWorker::worker_loop() {
    bind_to_core(cpu_core_);
    std::cout << "[CpuWorker] core" << cpu_core_ << " ready\n";

    while (true) {
        WorkerRequest req;
        {
            std::unique_lock<std::mutex> lk(q_mutex_);
            q_cv_.wait(lk, [this]{
                return !queue_.empty() || !running_.load();
            });
            if (!running_.load() && queue_.empty()) break;
            req = std::move(const_cast<WorkerRequest&>(queue_.top()));
            queue_.pop();
        }

        is_idle_.store(false);
        NodeInstance* ni = req.node_inst;

        if (ni->tmpl->node_type == NodeType::CPU) {
            // ---- 纯 CPU 节点（与 Day5 完全一致）----
            ni->state.store(NodeState::PRE_EXEC);
            ni->trace.t_pre_start_us = get_time_us();
            std::this_thread::sleep_for(
                std::chrono::microseconds(ni->tmpl->wcet_us));
            int64_t t1 = get_time_us();
            ni->trace.t_pre_end_us = t1;
            ni->trace.t_finish_us  = t1;
            ni->state.store(NodeState::COMPLETED);

            std::cout << "[CpuWorker] core" << cpu_core_
                      << " DONE(CPU) " << ni->job_id << "/" << ni->node_id
                      << " actual=" << (t1 - ni->trace.t_pre_start_us) << "us\n";

            mgr_->post_event({EventType::NODE_COMPLETE,
                              ni->job_id, ni->node_id, mgr_->now_us()});

        } else if (ni->tmpl->node_type == NodeType::GPU) {

            if (ni->state.load() == NodeState::DISPATCHED) {
                // ---- GPU 节点：第一阶段（CPU 提交工作）----
                ni->state.store(NodeState::PRE_EXEC);
                ni->trace.t_pre_start_us = get_time_us();

                // 模拟 CPU 提交工作：参数准备、buffer 分配等
                std::this_thread::sleep_for(
                    std::chrono::microseconds(ni->tmpl->pre_exec_us));

                // 注意：t_pre_end_us 由 handle_gpu_lock_request 记录
                // 确保时间戳在 RTMgr 侧统一记录，避免线程间时序偏差
                std::cout << "[CpuWorker] core" << cpu_core_
                          << " GPU_PRE_DONE " << ni->job_id << "/" << ni->node_id
                          << " → requesting GPU lock\n";

                // 请求 GPU 锁，然后直接返回（挂起让出 CPU）
                mgr_->post_event({EventType::GPU_LOCK_REQUEST,
                                  ni->job_id, ni->node_id, mgr_->now_us()});
                // 不发 RESOURCE_IDLE，让出 CPU 后立即可接受下一个任务

            } else if (ni->state.load() == NodeState::POST_EXEC) {
                // ---- GPU 节点：第二阶段（CPU 收尾工作）----
                // t_post_start_us 已在 handle_gpu_kernel_done 中记录
                std::this_thread::sleep_for(
                    std::chrono::microseconds(ni->tmpl->post_exec_us));

                int64_t t_post_end = get_time_us();
                ni->trace.t_post_end_us = t_post_end;
                ni->trace.t_finish_us   = t_post_end;
                ni->state.store(NodeState::COMPLETED);

                std::cout << "[CpuWorker] core" << cpu_core_
                          << " DONE(GPU_POST) " << ni->job_id << "/" << ni->node_id
                          << " post=" << (t_post_end - ni->trace.t_post_start_us)
                          << "us\n";

                mgr_->post_event({EventType::NODE_COMPLETE,
                                  ni->job_id, ni->node_id, mgr_->now_us()});
            }
        }

        // 当前阶段完成，通知 CPU 空闲（触发 try_dispatch 取下一个任务）
        is_idle_.store(true);
        mgr_->post_event({EventType::RESOURCE_IDLE, "", "", mgr_->now_us()});
    }
    std::cout << "[CpuWorker] core" << cpu_core_ << " exiting\n";
}

// ============================================================
// Dispatcher
// ============================================================
Dispatcher::Dispatcher(RuntimeManager* mgr) : mgr_(mgr) {}

void Dispatcher::register_cpu_worker(int core, std::shared_ptr<CpuWorker> w) {
    cpu_workers_[core] = std::move(w);
}