# C1 MuJoCo Sim2Sim

`deploy_c1.py` 使用本机 C1 训练 URDF 和训练 run 的 `params/env.yaml` 构建 MuJoCo 场景，支持 TorchScript 学生策略及 ONNX 学生策略、手柄控制，以及八种相连地形。

## 本机文件和环境

运行需要 Python 环境中的 `mujoco`、`numpy`、`scipy`、`torch`、`PyYAML`、`pygame`；使用 ONNX 时还需要 CPU `onnxruntime`。

本机另外提供：

- 导出的学生 `policy.pt` 或 `policy.onnx`。`model_*.pt` 原始训练检查点不能直接传给这个入口。
- 同一训练 run 的 `params/env.yaml`。
- C1 训练 URDF，以及 URDF 引用的全部 STL/碰撞网格。配置中的 `scene.robot.spawn.asset_path` 应指向本机 URDF 的有效绝对路径。

这些私有模型、URDF、网格和 `policy/` 部署包不包含在这次代码更新中。

## 全部地形和手柄

在仓库根目录激活本机 MuJoCo 环境，例如 `conda activate unitree_mujoco`，再运行：

```bash
bash deploy/deploy_mujoco/run_c1_sim2sim.sh \
  --policy /absolute/path/to/policy.onnx \
  --training-config /absolute/path/to/run/params/env.yaml
```

也可以直接运行 Python：

```bash
python deploy/deploy_mujoco/deploy_c1.py \
  --policy /absolute/path/to/policy.onnx \
  --training-config /absolute/path/to/run/params/env.yaml \
  --terrain all --difficulty 0.5 --stair-height 0.22 \
  --max-cmd 2.0 1.0 2.0
```

不传 `--cmd` 和 `--headless` 时打开窗口并使用手柄；未检测到手柄时提示并退出。左摇杆控制前后、横移，右摇杆控制转向。`Y` 或键盘 `R` 重置机器人及策略历史，关闭窗口退出。`--max-cmd` 的三个值分别是前后速度、横移速度及转向角速度上限。

## 八种地形与配色

八块 9×9 m 地形组成相连的 36×18 m 场地，机器人从平地出发。外围通道使用浅棕黄色，每种地形使用独立的柔和配色：

| 中心位置 | x=0 | x=9 | x=18 | x=27 |
| --- | --- | --- | --- | --- |
| y=9 | 粗糙坡：橄榄绿 | 上楼梯：玫瑰粉 | 下楼梯：淡紫 | 障碍：陶棕 |
| y=0 | 平地：沙黄 | 波浪：青绿 | 上坡：浅蓝 | 下坡：蜜桃色 |

颜色在 `deploy_c1.py` 的 `GROUND_RGBA` 和 `TERRAIN_RGBA` 中定义。地形块使用独立材质，台阶的每个碰撞盒也应用同一种地形颜色。其他调用 `load_terrain_park` 的入口可省略 `terrain_colors`，继续使用其原有材质。

`--difficulty` 范围为 `[0,1]`，默认 `0.5`。C1 的上下楼梯单级高度默认固定为 **0.22 m**，台阶宽度实际为 **0.30 m**。楼梯高度由 `--stair-height` 单独控制，不随 `--difficulty` 改变；其他地形继续按难度生成。水平网格为 0.1 m，垂直量化为 0.005 m，平滑地形使用高度场，楼梯和障碍使用合并盒柱保留垂直侧面。

`--terrain flat` 使用平地场景。`--terrain all` 保留各地形的实际高低起伏和凹坑，外围通道没有用无限平面填平楼梯区域。

## 无窗口检查

```bash
python deploy/deploy_mujoco/deploy_c1.py \
  --policy /absolute/path/to/policy.onnx \
  --training-config /absolute/path/to/run/params/env.yaml \
  --terrain all --stair-height 0.22 \
  --headless --duration 5 --cmd 0.5 0 0 \
  --report outputs/sim2sim/c1_smoke.json
```

`--cmd` 使用固定速度指令，并关闭手柄输入。此启动检查从平地出发，不等于通过全部八种地形。报告包含运行状态、楼梯高度、模拟时长、翻倒次数、数值异常和策略来源等信息。

## 学生策略接口

单帧观测为 57 维：角速度 3、投影重力 3、指令 3、关节位置 16、关节速度 16、上次动作 16。轮子的位置观测置零。

TorchScript 输入为单帧 `[1,57]`，导出的模型内部维护 4 帧历史。ONNX 输入 `obs` 为 float32 `[1,228]`，输出 `actions` 为 float32 `[1,16]`；适配器按项维护历史，项宽为 `[3,3,3,16,16,16]`，每项从旧帧到新帧，重置后首帧重复填充。

输出按原始 16 维动作处理：裁剪至 ±10，腿位置偏移缩放 0.25，轮速目标缩放 20 并限制在 ±150 rad/s。关节顺序为 FL、FR、RL、RR，每条腿依次为 hip、thigh、calf、foot。

地形公式适配自 Isaac Lab；许可及声明见 [ISAACLAB_LICENSE.txt](ISAACLAB_LICENSE.txt)。
