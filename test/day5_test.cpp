// src/day5_test.cpp
#include "task_model.h"
#include "runtime_manager.h"
#include "scheduler.h"
#include "dispatcher.h"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== Day 5 Test: 多队列调度 + CPU Worker 绑核 ===\n\n";
    std::cout << "预期：TASK_A(p=1) 的节点优先于 TASK_B(p=2) 被 DISPATCH\n\n";

    TaskTemplate tmpl_a = build_task_a();
    TaskTemplate tmpl_b = build_task_b();

    RuntimeManager mgr(0);
    auto sched = std::make_shared<BaselineScheduler>();
    auto disp  = std::make_shared<Dispatcher>(&mgr);

    // core1 对应 resource_id=1（TASK_A），core2 对应 resource_id=2（TASK_B）
    auto w1 = std::make_shared<CpuWorker>(1, &mgr);
    auto w2 = std::make_shared<CpuWorker>(2, &mgr);
    disp->register_cpu_worker(1, w1);
    disp->register_cpu_worker(2, w2);

    mgr.set_scheduler(sched);
    mgr.set_dispatcher(disp);

    mgr.start();
    w1->start();
    w2->start();

    // 同时释放两个 DAG，各 2 个周期
    ReleaseTimer ta(&mgr, &tmpl_a, 2);
    ReleaseTimer tb(&mgr, &tmpl_b, 2);
    ta.start();
    tb.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    ta.stop(); tb.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    w1->stop(); w2->stop();
    mgr.stop();

    // ---- 今日验收（对照日志）----
    // 1. [Sched] ENQUEUE：A节点进 resource1 队列，B节点进 resource2 队列
    // 2. [RTMgr] try_dispatch：每个 Worker 空闲时取自己队列的队头
    // 3. [CpuWorker] DONE：A节点全在 core1，B节点全在 core2
    // 4. 所有 job 打印 JOB_COMPLETE
    // 注意：由于 A 和 B 各自使用独立队列，本测试中优先级的作用体现在
    //       同一个核上有多个节点排队时（如 TASK_A 连续释放两个 job）
    std::cout << "\n[Day5 验收] A节点→core1，B节点→core2，"
              << "同核多节点时高优先级先执行\n";
    return 0;
}