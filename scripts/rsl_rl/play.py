# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""使用 RSL-RL 训练好的检查点进行推理播放的脚本。"""

"""首先启动 Isaac Sim 仿真器。"""

import argparse
import os
import sys

# 确保本地导入优先于 pip 包（例如 cv2.utils）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 在导入 isaacsim/Kit 模块前预加载原生扩展，避免后续 Isaac Lab/RSL-RL 导入时
# 发生 Windows 动态链接库（DLL）加载冲突。
import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip
from rsl_rl_utils import export_cts_policy_as_jit, export_cts_policy_as_onnx

# 添加命令行参数
parser = argparse.ArgumentParser(description='Train an RL agent with RSL-RL.')
parser.add_argument('--video', action='store_true', default=False, help='Record videos during training.')
parser.add_argument('--video_length', type=int, default=int(1e9), help='Length of the recorded video (in steps).')
parser.add_argument(
    '--disable_fabric', action='store_true', default=False, help='Disable fabric and use USD I/O operations.'
)
parser.add_argument('--num_envs', type=int, default=None, help='Number of environments to simulate.')
parser.add_argument('--task', type=str, default=None, help='Name of the task.')
parser.add_argument(
    '--agent', type=str, default='rsl_rl_cfg_entry_point', help='Name of the RL agent configuration entry point.'
)
parser.add_argument('--seed', type=int, default=None, help='Seed used for the environment')
parser.add_argument(
    '--use_pretrained_checkpoint',
    action='store_true',
    help='Use the pre-trained checkpoint from Nucleus.',
)
parser.add_argument('--real-time', action='store_true', default=False, help='Run in real-time, if possible.')
parser.add_argument('--keyboard', action='store_true', default=False, help='Whether to use keyboard.')
parser.add_argument('--fix_commands', action='store_true', default=False, help='Fix the velocity commands.')
# 追加 RSL-RL 命令行参数
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli, hydra_args = parser.parse_known_args()
# 录制视频时必须启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清空 sys.argv 以交给 Hydra 处理
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余逻辑从此处开始。"""

import gymnasium as gym
import time
import torch
from tensordict import TensorDict
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerCTS

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import robot_lab.tasks  # noqa: F401

def fix_commands(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """将速度指令固定为恒定目标值。

    优先锁定指令生成器，使策略观测与环境内部指令状态保持一致。

    Args:
        env_cfg: 环境配置。
    """
    fixed_lin_x, fixed_lin_y, fixed_ang_z = 1.0, 0.0, 0.0

    base_velocity_cfg = getattr(getattr(env_cfg, 'commands', None), 'base_velocity', None)
    if base_velocity_cfg is None:
        return

    fixed_cfg = UniformVelocityCommandCfg(
        asset_name=getattr(base_velocity_cfg, 'asset_name', 'robot'),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        resampling_time_range=(5.0, 5.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(fixed_lin_x, fixed_lin_x),
            lin_vel_y=(fixed_lin_y, fixed_lin_y),
            ang_vel_z=(fixed_ang_z, fixed_ang_z),
            heading=None,
        ),
        debug_vis=True,
    )
    env_cfg.commands.base_velocity = fixed_cfg

    # terrain_levels_vel_gym 依赖自定义 Go2RLGymCommand 字段
    if hasattr(env_cfg, 'curriculum') and hasattr(env_cfg.curriculum, 'terrain_levels'):
        env_cfg.curriculum.terrain_levels = None

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """使用 RSL-RL 智能体执行推理播放。"""
    # 获取任务名以构造检查点路径
    task_name = args_cli.task.split(':')[-1]

    # 使用非 Hydra 命令行参数覆盖配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # 设置环境种子
    # 注意：部分随机化在环境初始化时发生，因此在此处设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 推理时禁用随机化
    env_cfg.observations.policy.enable_corruption = False
    # 移除随机推力
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    # 指定实验日志根目录
    log_root_path = os.path.join('logs', 'rsl_rl', agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f'[INFO] Loading experiment from directory: {log_root_path}')
    if args_cli.use_pretrained_checkpoint:
        # resume_path = get_published_pretrained_checkpoint('rsl_rl', task_name)
        # if not resume_path:
        #     print('[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.')
        #     return
        raise NotImplementedError('Pre-trained checkpoint retrieval is disabled temporarily.')
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 为环境设置日志目录（适用于所有环境类型）
    env_cfg.log_dir = log_dir

    # 若指定则固定速度指令
    if args_cli.fix_commands:
        fix_commands(env_cfg)

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode='rgb_array' if args_cli.video else None)

    # 若 RL 算法需要则转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装为视频录制环境
    if args_cli.video:
        import imageio
        video_path = os.path.join(log_dir, 'videos', 'play', time.strftime('%Y-%m-%d_%H-%M-%S') + '.mp4')
        writer = imageio.get_writer(video_path, fps=int(1 / env.unwrapped.step_dt))
        # video_kwargs = {
        #     'video_folder': os.path.join(log_dir, 'videos', 'play'),
        #     'step_trigger': lambda step: step == 0,
        #     'video_length': args_cli.video_length,
        #     'disable_logger': True,
        # }
        # print('[INFO] Recording videos during training.')
        # print_dict(video_kwargs, nesting=4)
        # 将所有帧存入列表速度慢且占用内存，因此使用 imageio
        # env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 包装为 RSL-RL 向量环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f'[INFO]: Loading model checkpoint from: {resume_path}')
    # 加载已训练模型
    if agent_cfg.class_name == 'OnPolicyRunner':
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == 'DistillationRunner':
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == 'OnPolicyRunnerCTS':
        runner = OnPolicyRunnerCTS(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f'Unsupported runner class: {agent_cfg.class_name}')
    runner.load(resume_path)

    # 获取训练好的策略用于推理
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 提取神经网络模块
    # 使用 try-except 以保持向后兼容
    try:
        # 2.3 及以上版本
        policy_nn = runner.alg.policy
    except AttributeError:
        # 2.2 及以下版本
        policy_nn = runner.alg.actor_critic

    # 提取归一化器
    if hasattr(policy_nn, 'actor_obs_normalizer'):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, 'student_obs_normalizer'):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # 导出策略为 onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), 'exported')
    if agent_cfg.class_name == 'OnPolicyRunnerCTS':
        export_cts_policy_as_jit(policy_nn, actor_obs_normalizer=policy_nn.actor_obs_normalizer, single_obs_normalizer=policy_nn.single_obs_normalizer, path=export_model_dir, filename='policy.pt')
        export_cts_policy_as_onnx(policy_nn, actor_obs_normalizer=policy_nn.actor_obs_normalizer, single_obs_normalizer=policy_nn.single_obs_normalizer, path=export_model_dir, filename='policy.onnx')
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename='policy.pt')
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename='policy.onnx')

    def _get_obs_dict(obs_result):
        """从环境获取完整观测字典。

        兼容旧版 IsaacLab API（get_observations() 返回 TensorDict）
        与新版 API（返回元组 (policy_obs, extras)）。
        若观测为普通字典则转换为 TensorDict。

        Args:
            obs_result: 环境返回的观测结果。

        Returns:
            环境观测的 TensorDict。
        """
        if isinstance(obs_result, tuple):
            # 新版 IsaacLab API: (policy_obs, {'observations': full_obs_dict})
            obs = obs_result[1]['observations']
        else:
            obs = obs_result
        if not isinstance(obs, TensorDict):
            # 从第一个观测张量推断 batch size
            batch_size = [next(iter(obs.values())).shape[0]]
            obs = TensorDict(obs, batch_size=batch_size, device=env.unwrapped.device)
        return obs

    dt = env.unwrapped.step_dt

    # env.unwrapped.eye = (1.1, 3.3, 0.9)
    # 重置环境
    obs = _get_obs_dict(env.get_observations())
    timestep = 0
    # 运行环境仿真
    while simulation_app.is_running():
        start_time = time.time()
        # 推理模式下运行
        with torch.inference_mode():
            # 智能体动作
            actions = policy(obs)
            # 环境步进
            _, _, dones, extras = env.step(actions)
            obs = _get_obs_dict(extras['observations'])
            # 重置已终止 episode 的循环状态
            policy_nn.reset(dones)
        if args_cli.video:
            writer.append_data(env.env.render())

        # 实时评估时的时间延迟
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭仿真器
    env.close()

    # 关闭视频写入器
    if args_cli.video:
        writer.close()

if __name__ == '__main__':
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
