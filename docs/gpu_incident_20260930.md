# GPU 训练故障与整机卡死记录（2026-09-30）

## 当前状态

用户报告诊断期间整机卡死后，停止所有 GPU 训练、仿真与压力测试。
16:19 检查未发现 physics_probe、compute-sanitizer、gpu_compute_probe 或 GPU 计算进程。
没有更改系统驱动、内核、BIOS、启动参数或 w1w_deploy。
Isaac Sim 5.1 对照没有有效结果，不能称其已解决问题。

## 已确认的三个故障现象

1. NVIDIA / CUDA：原训练及官方 Go2 的物理仿真都出现 CUDA 700，内核记录 Xid 13/31/43，另一次出现 Xid 69 和 CUDA 719。故障不局限于本项目 URDF 或 PPO。不能据此断定显卡硬件损坏，也不能保证降低为 2048 环境即可解决。
2. 系统硬锁死：09:12 启动的系统在 16:05:35 记录 `watchdog: CPU19: Watchdog detected hard LOCKUP on cpu 19`，没有保存后续调用栈。它出现在隔离版本诊断期间，但日志不足以确定是哪个驱动、线程或硬件导致。
3. Intel 核显：16:09:59 重启后，16:10:17、16:12:42、16:13:05–08 连续出现 `i915 ... GUC: Exception notification`、`GPU HANG`、`Resetting chip for dead GuC`。这是独立可见的显示驱动故障证据，不能将其直接归因于 RTX 5080 的 CUDA 故障。

## 已做测试的边界

- PyTorch 计算与约 6 GB 显存测试通过 180 秒；小内核突发测试通过 60 秒。这些短测试不能排除硬件故障。
- 临时加载 Warp 1.8.1 后 UUID 警告消失，官方 Go2 一次 3000 步通过；实际 2048 环境训练仍在第 166 轮 CUDA 700。因此 Warp 不是已验证的修复。
- Compute Sanitizer 停在初始化，未得到有效检查结果。
- 已有 Isaac Sim 5.1 环境的对照被系统重启打断；本次日志为空，未验证兼容性或稳定性。
- 以上没有覆盖安装现有 Conda/Isaac 环境；Warp 临时安装此前位于 /tmp。

## 后续工作限制

优先处理系统与显示稳定性，暂不重复启动 GPU 负载。
更换内核、固件、驱动或显示输出涉及系统级变更及可能的重启，应先与用户确认具体方案。
本机当前内核 7.0.0-31，另有 7.0.0-30；不能因其版本号较旧就宣称其稳定或会修复故障。
保留现有训练设置，暂不继续通过修改 URDF 或奖励项猜测修复内核故障。

## 原始证据

- `logs/gpu_diagnosis/kernel_before_fix.log`：此前 NVIDIA Xid。
- `logs/gpu_diagnosis/kernel_hard_lockup_160535.log`：CPU 硬锁死。
- `logs/gpu_diagnosis/kernel_boot_160959.log`：重启后 Intel 核显反复挂起。
- `logs/gpu_diagnosis/w1w_warp181_2048_train.log`：Warp 对照训练失败。

参考：[Linux i915 驱动文档](https://docs.kernel.org/next/gpu/i915.html)。
