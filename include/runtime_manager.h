// include/runtime_manager.h（Day5版）
#pragma once
#include "task_model.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <memory>

enum class EventType {
    JOB_RELEASE,
    NODE_COMPLETE,
    RESOURCE_IDLE,
    GPU_LOCK_REQUEST,   // ← 新增：CPU 提交完成，请求 GPU 锁
    GPU_KERNEL_DONE,    // ← 新增：GPU kernel 执行完成
    SHUTDOWN
};

struct RuntimeEvent {
    EventType           type;
    std::string         job_id;
    std::string         node_id;
    int64_t             timestamp_us  = 0;
    const TaskTemplate* task_tmpl     = nullptr;
};

// GPU 队列条目
struct GpuQueueEntry {
    NodeInstance* node_inst;
    int           task_priority;
    int64_t       queue_enter_us;
    bool operator>(const GpuQueueEntry& o) const {
        if (task_priority != o.task_priority)
            return task_priority > o.task_priority;
        return queue_enter_us > o.queue_enter_us;
    }
};

std::vector<NodeInstance*> get_initial_ready_nodes(JobInstance&);
std::vector<NodeInstance*> advance_dag(JobInstance&, const std::string&, int64_t);

class BaselineScheduler;
class Dispatcher;
class GpuExecutor;

class RuntimeManager {
public:
    explicit RuntimeManager(int core = 0);
    ~RuntimeManager();

    void post_event(RuntimeEvent evt);
    void start();
    void stop();
    int64_t now_us() const;

    void set_scheduler(std::shared_ptr<BaselineScheduler> s);
    void set_dispatcher(std::shared_ptr<Dispatcher> d);
    // 新增接口
    void set_gpu_executor(std::shared_ptr<GpuExecutor> ge);

private:
    void event_loop();
    void handle_job_release(const RuntimeEvent&);
    void handle_node_complete(const RuntimeEvent&);
    void handle_resource_idle(const RuntimeEvent&);

    // 遍历所有 CPU Worker，将空闲 Worker 对应队列的队头节点提交
    void try_dispatch();

    // 根据节点类型路由：CPU 节点入 Scheduler 队列，GPU 节点直接 submit
    void enqueue_node(NodeInstance* ni);

    int                  core_;
    std::thread          thread_;
    std::atomic<bool>    running_{false};

    std::queue<RuntimeEvent>   eq_;
    std::mutex                 eq_mutex_;
    std::condition_variable    eq_cv_;

    std::unordered_map<std::string, std::unique_ptr<JobInstance>> active_jobs_;
    std::mutex jobs_mutex_;

    int64_t          start_time_us_;
    std::atomic<int> jobs_released_{0};
    std::atomic<int> jobs_completed_{0};

    std::shared_ptr<BaselineScheduler>  scheduler_;
    std::shared_ptr<Dispatcher>         dispatcher_;
    // 以下是GPU相关
    std::shared_ptr<GpuExecutor>        gpu_executor_;

    std::priority_queue<GpuQueueEntry,
                        std::vector<GpuQueueEntry>,
                        std::greater<GpuQueueEntry>> gpu_queue_;
    std::mutex        gpu_queue_mutex_;
    std::atomic<bool> gpu_lock_{false};   // false=空闲，true=被占用

    
    // 新增事件处理
    void handle_gpu_lock_request(const RuntimeEvent&);
    void handle_gpu_kernel_done(const RuntimeEvent&);

    // 辅助：从 active_jobs_ 中找到节点实例指针
    NodeInstance* find_node_instance(const std::string& job_id,
                                    const std::string& node_id);
};

class GpuExecutor {
public:
    explicit GpuExecutor(RuntimeManager* mgr) : mgr_(mgr) {}

    // 异步提交 kernel，在独立线程中执行，完成后发 GPU_KERNEL_DONE
    // 不阻塞 RuntimeManager 事件循环
    void submit_kernel(NodeInstance* ni) {
        std::thread([this, ni]() {
            // Mock：sleep 模拟 kernel 执行时间
            // Day 9 替换为真实 OpenCL clEnqueueNDRangeKernel + clWaitForEvents
            std::this_thread::sleep_for(
                std::chrono::microseconds(ni->tmpl->accel_exec_us));

            mgr_->post_event({EventType::GPU_KERNEL_DONE,
                              ni->job_id, ni->node_id,
                              mgr_->now_us()});
        }).detach();
    }

private:
    RuntimeManager* mgr_;
};

class ReleaseTimer {
public:
    ReleaseTimer(RuntimeManager* mgr, const TaskTemplate* tmpl, int num_periods = 3);
    void start();
    void stop();
private:
    void run();
    RuntimeManager*      mgr_;
    const TaskTemplate*  tmpl_;
    int                  num_periods_;
    std::thread          thread_;
    std::atomic<bool>    running_{false};
    int                  counter_{0};
};