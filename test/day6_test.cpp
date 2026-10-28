#include "task_model.h"
#include "runtime_manager.h"
#include "scheduler.h"
#include "dispatcher.h"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

TaskTemplate build_task_a();
TaskTemplate build_task_b();

int main() {
    std::cout << "=== Day 6 Test: GPU 锁队列 + 节点挂起让出 CPU ===\n\n";
    std::cout << "预期：\n"
              << "  A3: GPU_LOCK 立即取到，queue_wait=0\n"
              << "  B3: GPU_LOCK 排队等待，queue_wait > 0\n"
              << "  A3/B3 GPU 执行段不重叠\n"
              << "  A3/B3 的 CPU 前处理和后处理均在绑定核上完成\n\n";

    TaskTemplate tmpl_a = build_task_a();
    TaskTemplate tmpl_b = build_task_b();

    RuntimeManager mgr(0);
    auto sched    = std::make_shared<BaselineScheduler>();
    auto disp     = std::make_shared<Dispatcher>(&mgr);
    auto gpu_exec = std::make_shared<GpuExecutor>(&mgr);

    auto w1 = std::make_shared<CpuWorker>(1, &mgr);
    auto w2 = std::make_shared<CpuWorker>(2, &mgr);
    disp->register_cpu_worker(1, w1);
    disp->register_cpu_worker(2, w2);

    mgr.set_scheduler(sched);
    mgr.set_dispatcher(disp);
    mgr.set_gpu_executor(gpu_exec);

    mgr.start();
    w1->start();
    w2->start();

    // TASK_B 延迟 20ms，制造 GPU 竞争
    // 时序：
    //   t=0:    A1 就绪，core1 执行 A1(5ms)
    //   t=5ms:  A3 就绪，core1 执行 A3 前处理(1ms)，请求 GPU 锁 → 立即取到
    //   t=6ms:  GPU 开始执行 A3 kernel(13ms)，core1 空闲可接受 A2
    //   t=20ms: B1 就绪，core2 执行 B1(8ms)
    //   t=28ms: B3 就绪，core2 执行 B3 前处理(1ms)，请求 GPU 锁 → 排队等待
    //           B3.queue_wait ≈ (6+13) - 29 = 时序差值
    //   t=19ms: A3 GPU 完成，A3 收尾进 core1 队列
    //   GPU 锁给 B3，B3.queue_wait 可观测
    ReleaseTimer ta(&mgr, &tmpl_a, 2);
    ReleaseTimer tb(&mgr, &tmpl_b, 2);
    ta.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    tb.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    ta.stop(); tb.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    w1->stop(); w2->stop();
    mgr.stop();

    // ---- 今日验收（对照日志）----
    // 1. [RTMgr] GPU_LOCK granted immediately A3/...
    // 2. [RTMgr] GPU_LOCK queued B3/...
    // 3. [RTMgr] GPU_LOCK → B3/... queue_wait=xxxus
    // 4. [RTMgr] GPU_KERNEL_DONE A3 先于 GPU_KERNEL_DONE B3
    // 5. [CpuWorker] core1 DONE(GPU_POST) A3（收尾在 core1）
    // 6. [CpuWorker] core2 DONE(GPU_POST) B3（收尾在 core2）
    // 7. 所有 JOB_COMPLETE 均打印
    std::cout << "\n[Day6 验收] 对照以上 7 条日志项 → 全部出现则 PASS\n";
    return 0;
}