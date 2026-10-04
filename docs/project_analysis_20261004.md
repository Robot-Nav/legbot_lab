# 项目分析与 GitHub 同步说明（2026-10-04）

## 总体结构

本项目是面向 16 自由度轮足机器人的强化学习训练、策略导出、仿真验证与实机部署代码库。训练侧依赖 Isaac Lab，算法侧使用仓库内定制的 RSL-RL，部署侧包含 MuJoCo 验证和独立 W1W 控制服务。

| 目录 | 作用 | 主要入口 |
| --- | --- | --- |
| `source/robot_lab` | 机器人资产、Gym 任务注册、观测、动作、奖励、随机化和课程 | `robot_lab/tasks/legbot/env_cfg.py` |
| `source/rsl_rl` | PPO、CTS 与 MoE 网络、rollout 存储、训练器和策略导出 | `rsl_rl/algorithms/moe_cts.py` |
| `scripts/rsl_rl` | 训练、播放、导出、错误捕获与运行库诊断 | `train.py`、`play.py` |
| `scripts/diagnostics` | GPU/物理故障诊断、按速度筛选并导出检查点 | 诊断脚本和 `select_export_speed_policies.py` |
| `deploy/deploy_mujoco` | 策略的 MuJoCo 仿真验证 | `deploy_legbot.py` |
| `w1w_deploy` | Python 策略控制器、C++ 电机/IMU/ELRS 服务、systemd、安装与回滚脚本、网页监控 | `controller/deploy.py` |
| `resources` | Go2 参考资产以及本地私有 W1W/Legbot 资产 | 私有目录不上传 |
| `tests` | 训练接口、算法更新、碰撞过滤和本地模型检查 | unittest |
| `docs` | 训练审查、模型审计和 GPU 故障记录 | 各审查文档 |

## 训练到部署的数据链路

1. `RobotLab-Legbot-v0` 与 `RobotLab-Legbot-Simplified-v0` 使用 W1W 简化模型；`RobotLab-Legbot-Original-v0` 提供原始网格模型对照任务。
2. 机器人动作按四条腿依次排列，每条腿为髋、大腿、小腿、轮，合计 16 维；腿使用位置控制，轮使用速度控制。
3. 单帧本体观测为 57 维，四帧历史为 228 维。轮关节位置使用零占位；历史按观测项分组编码，导出和部署需要保持相同布局。
4. 教师编码器从特权观测得到 32 维隐变量；学生使用 8 个 MoE 专家从历史观测估计隐变量。Actor 和 Critic 共享，学生编码器通过独立优化器进行蒸馏与门控负载均衡训练。
5. 策略导出保留学生路径。MuJoCo 和实机控制器均需遵循一致的关节顺序、观测缩放、默认站姿与动作缩放。

当前 Legbot 默认配置：2048 个环境、教师比例 0.75、每轮 24 步、5 个 PPO epoch、4 个 mini-batch；物理步长 0.005 秒，控制周期 0.02 秒，回合上限 25 秒；每 50 轮保存，默认最大训练轮数 300000。命令行覆盖后的实际配置以运行目录中的参数文件为准。

## 部署代码检查

W1W 部署包包含 12 个 RobStride 关节与 4 个 W190 轮电机接口，使用本地 UDP 交换电机、IMU 与遥控数据。配置类检查 16 维动作、57 维观测、轮索引和控制期限等约束。

代码和现有测试覆盖阻尼数据包、反馈顺序与命令回显、遥控急停、过期反馈锁存、控制发送期限，以及 systemd 控制服务不自动重启等行为。安装和策略启动为独立步骤。此次分析只执行单元测试与 CRSF 解析器测试，没有启动实机控制器、电机服务或 GPU 仿真。

## 本次验证结果

| 检查 | 结果 |
| --- | --- |
| 上传候选 Python 文件 AST 语法检查 | 全部通过 |
| 根目录 unittest，CPU 运行 | 22 项：17 通过，3 跳过，1 失败，1 错误 |
| W1W 部署 unittest | 32 项全部通过 |
| C++ CRSF 解析器测试 | 编译并运行通过 |

根目录测试使用现有 `rpo_isaaclab` Python 环境并设置 `CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=source/rsl_rl`。3 项跳过测试依赖未配置到此 Python 路径的 USD 运行库。

两项未通过的检查均涉及用户要求不上传的本地私有模型：

- `test_original_physics_and_joint_interface`：简化与原始 URDF 的惯性元素不一致，比较中发现原始模型存在的惯性 `origin` 在另一模型中缺失。
- `test_urdf_mujoco_parity`：某个 URDF 惯性元素缺少 `origin`，测试读取时发生 `AttributeError`，因此不能宣称 URDF 与 MuJoCo 惯性参数完全一致。

上述问题原样记录，未修改私有模型或放宽测试断言。仓库不包含私有资产时，这些模型检查按现有测试逻辑跳过。

部署测试从 `w1w_deploy` 目录运行；受限沙箱禁止创建 socket，故在允许本地 socket 的执行环境重跑后全部通过。测试仅构造协议对象和本地反馈数据。

## 使用边界与待处理事项

- 私有模型与策略权重不随仓库分发，克隆后若要训练 Legbot 或部署 W1W，需自行在配置指定位置提供合法模型和权重。
- 本地 GPU 故障记录见 `gpu_incident_20260930.md`，本次没有进行 GPU 训练、压力测试或驱动修改，CPU 测试结果不能证明 GPU 长期训练稳定。
- 本次没有在干净环境中完整安装 Isaac Lab，也没有执行实机验证；README 中的依赖版本不能据此视为已完成端到端兼容性验证。
- `source/robot_lab/setup.py` 显式只列出顶层包 `robot_lab`，若要分发非 editable wheel，应单独检查任务和资产子包是否被打包；本次保留现有安装方式与源码。
- 根 README 引用的两张 `resources/results` 图片本地已删除；本次同步保留该删除状态，相应图片引用可能无法展示。

## 上传范围

同步目标为 `Robot-Nav/legbot_lab` 的 `WF-CTS-MOE` 分支。以当前工作目录中的实际文件为准，覆盖该分支对应源码和文档，并保留远端基础提交历史。

排除项：

- `resources/w1w_wf/` 与 `resources/legbot_wf/` 全目录，包括 URDF、网格、MuJoCo 场景和派生文件。
- ONNX、PTH、PT、CKPT 等模型/策略文件，以及已有忽略规则覆盖的数据文件。
- `w1w_129_build.zip`、`w1w_urdf1.zip` 和打包运行归档。
- `logs/`、`outputs/`、本地虚拟环境、缓存、编译产物和机器凭据目录。

Go2/Go2W 参考模型继续保留。新增模型压缩包忽略规则，防止后续普通 `git add` 意外上传。使用独立临时克隆创建上传提交，原工作目录的 Git 暂存区与未提交改动保持原状；本次仅在原目录补充 `.gitignore` 和此报告。
