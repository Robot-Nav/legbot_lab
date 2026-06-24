# Vbot RL Lab

基于 NVIDIA Isaac Lab 框架的四足机器人 Vbot 强化学习训练与部署项目，实现从仿真训练到实物部署的完整闭环（Sim2Real）。

## 项目结构

```
vbot_rl_lab/
├── source/unitree_rl_lab/          # Python 训练源码（基于 unitree_rl_lab 框架）
│   └── unitree_rl_lab/
│       ├── assets/robots/          # Vbot 机器人配置
│       │   ├── unitree.py          # UNITREE_VBOT_CFG（URDF 加载 + 执行器配置）
│       │   └── unitree_actuators.py # Vbot 执行器 T-N 曲线模型
│       ├── tasks/locomotion/       # 速度跟踪运动任务
│       │   ├── agents/             # PPO 算法配置
│       │   ├── mdp/                # MDP 组件（观测/奖励/课程学习/命令）
│       │   └── robots/vbot/        # Vbot 环境配置
│       └── utils/                  # 工具函数（部署配置导出）
├── scripts/                        # 训练/推理脚本
│   └── rsl_rl/
│       ├── train.py                # 训练入口（支持 headless 模式）
│       ├── play.py                 # 推理/ONNX导出入口
│       └── cli_args.py             # 命令行参数
├── deploy/                         # C++ 部署代码
│   └── robots/vbot/                # Vbot 部署程序
│       ├── config/config.yaml      # FSM 状态机配置
│       ├── include/                # 部署类型定义
│       ├── src/                    # RL 状态逻辑
│       └── CMakeLists.txt          # 部署构建配置
├── unitree_ros/                     # 机器人模型文件
│   └── robots/vbot_description/
│       ├── urdf/vbot_description.urdf  # Vbot URDF（无 mesh 依赖，纯几何体）
│       └── meshes/vbot.xml             # 原始 MJCF 定义
└── outputs/                        # 训练输出（Hydra 日志 & 配置）
```

## 准备工作

### 1. 安装 Isaac Lab

严格按照 [Isaac Lab 官方指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 完成安装。

### 2. 克隆并安装本项目

```bash
# 激活 Isaac Lab 的 conda 环境
conda activate env_isaaclab

# 安装本项目为 editable 模式
pip install -e source/unitree_rl_lab
```

### 3. Vbot 模型文件

Vbot 的 URDF 模型文件已内置于项目中，位于：

```
unitree_ros/robots/vbot_description/urdf/vbot_description.urdf
```

该 URDF 使用纯几何体（box/cylinder/sphere），无需额外下载 mesh 文件。

原始 MJCF 模型定义位于：

```
unitree_ros/robots/vbot_description/meshes/vbot.xml
```

## Vbot 机器人参数

| 参数 | 值 |
|------|-----|
| 站立高度 | 0.462 m |
| 基座质量 | 9.016 kg |
| 自由度 | 12（四足） |
| 髋/大腿电机力矩 | ±17 N·m |
| 小腿电机力矩 | ±34 N·m |

## 开始训练 Vbot

### 无头模式训练（推荐）

```bash
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --headless
```

可选参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_envs` | 并行环境数 | 4096 |
| `--max_iterations` | 最大训练迭代 | 50000 |
| `--seed` | 随机种子 | 随机 |

完整示例：

```bash
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --headless --num_envs 4096 --max_iterations 15000
```

### 恢复中断的训练

```bash
# 自动查找最新检查点恢复
python scripts/rsl_rl/train.py --task Unitree-Vbot-Velocity --resume --headless

# 手动指定实验目录和检查点
python scripts/rsl_rl/train.py \
  --task Unitree-Vbot-Velocity \
  --resume \
  --load_run 2026-05-15_10-00-00 \
  --checkpoint model_108.pt \
  --headless
```

> 检查点文件存放在 `logs/rsl_rl/unitree_vbot_velocity/<run_name>/` 目录下。

### 使用已训练模型进行推理演示

```bash
python scripts/rsl_rl/play.py --task Unitree-Vbot-Velocity
```

推理时会自动导出 ONNX 模型（`policy.onnx`），用于后续的 C++ 部署。

## 部署到实物机器人

训练完成后，进行 **Sim2Sim（仿真验证）** 和 **Sim2Real（实物部署）**：

1. **安装依赖**：安装 `unitree_sdk2`。
2. **编译控制器**：
   ```bash
   cd deploy/robots/vbot
   mkdir build && cd build && cmake .. && make
   ```
3. **执行 Sim2Sim**：在 MuJoCo 中验证训练好的 ONNX 策略。
4. **执行 Sim2Real**：运行编译好的 Vbot 控制器程序，指定网络接口连接真实机器人。

## 技术栈

| 类别 | 技术 |
|------|------|
| 仿真平台 | NVIDIA Isaac Sim 5.1.0 / Isaac Lab 2.3.0 |
| RL 算法 | RSL-RL 2.3.1（PPO） |
| 深度学习 | PyTorch（CUDA 加速） |
| 模型导出 | ONNX Runtime 1.22.0 |
| 部署语言 | C++17 |
| 机器人模型 | URDF（纯几何体，无 mesh 依赖） |
| 配置管理 | Hydra / YAML |