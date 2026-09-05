# Copyright (c) 2026 Robot-Nav
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""使用 RSL-RL 训练 RL 智能体的脚本。"""

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

# 添加命令行参数
parser = argparse.ArgumentParser(description='Train an RL agent with RSL-RL.')
parser.add_argument('--video', action='store_true', default=False, help='Record videos during training.')
parser.add_argument('--video_length', type=int, default=200, help='Length of the recorded video (in steps).')
parser.add_argument('--video_interval', type=int, default=2000, help='Interval between video recordings (in steps).')
parser.add_argument('--num_envs', type=int, default=None, help='Number of environments to simulate.')
parser.add_argument('--task', type=str, default=None, help='Name of the task.')
parser.add_argument(
    '--agent', type=str, default='rsl_rl_cfg_entry_point', help='Name of the RL agent configuration entry point.'
)
parser.add_argument('--seed', type=int, default=None, help='Seed used for the environment')
parser.add_argument('--max_iterations', type=int, default=None, help='RL Policy training iterations.')
parser.add_argument(
    '--distributed', action='store_true', default=False, help='Run training with multiple GPUs or nodes.'
)
parser.add_argument('--export_io_descriptors', action='store_true', default=False, help='Export IO descriptors.')
# 追加 RSL-RL 命令行参数
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录制视频时必须启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清空 sys.argv 以交给 Hydra 处理
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""检查最低支持的 RSL-RL 版本。"""

import importlib.metadata as metadata

from packaging import version

# 检查最低 rsl-rl 版本
RSL_RL_VERSION = '3.0.1'
installed_version = metadata.version('rsl-rl-lib')
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [r'python', '-m', 'pip', 'install', f'rsl-rl-lib=={RSL_RL_VERSION}']
    cmd_str = ' '.join(cmd)
    print(
        f'Please install the correct version of RSL-RL.\nExisting version is: \'{installed_version}\''
        f' and required version is: \'{RSL_RL_VERSION}\'.\nTo install the correct version, run:'
        f'\n\n\t{cmd_str}\n'
    )
    exit(1)

"""其余逻辑从此处开始。"""

import gymnasium as gym
import torch
from datetime import datetime

# 本地导入
from rsl_rl_utils import Logger

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerCTS

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """使用 RSL-RL 智能体执行训练。"""
    # 使用非 Hydra 命令行参数覆盖配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict['robogauge'] = {
        'enabled': args_cli.robogauge,
        'port': args_cli.robogauge_port,
    }
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境种子
    # 注意：部分随机化在环境初始化时发生，因此在此处设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # 检查 CPU 设备与分布式训练的不兼容组合
    if args_cli.distributed and args_cli.device is not None and 'cpu' in args_cli.device:
        raise ValueError(
            'Distributed training is not supported when using CPU device. '
            'Please use GPU device (e.g., --device cuda) for distributed training.'
        )

    # 多 GPU 训练配置
    if args_cli.distributed:
        env_cfg.sim.device = f'cuda:{app_launcher.local_rank}'
        agent_cfg.device = f'cuda:{app_launcher.local_rank}'

        # 为不同进程设置不同种子以保证多样性
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 指定实验日志根目录
    log_root_path = os.path.join('logs', 'rsl_rl', agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f'[INFO] Logging experiment in directory: {log_root_path}')
    # 指定运行日志目录：{时间戳}_{运行名}
    # Ray Tune 工作流通过以下日志行提取实验名，请勿修改（参见 PR #2346, comment-2819298849）
    log_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f'Exact experiment name requested from command line: {log_dir}')
    if agent_cfg.run_name:
        log_dir += f'_{agent_cfg.run_name}'
    log_dir = os.path.join(log_root_path, log_dir)

    # 若请求则设置 IO 描述符导出标志
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            'IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported.'
        )

    # 为环境设置日志目录（适用于所有环境类型）
    env_cfg.log_dir = log_dir

    # 创建 Isaac 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode='rgb_array' if args_cli.video else None)

    # 若 RL 算法需要则转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 在创建新 log_dir 前保存恢复路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == 'Distillation':
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 包装为视频录制环境
    if args_cli.video:
        video_kwargs = {
            'video_folder': os.path.join(log_dir, 'videos', 'train'),
            'step_trigger': lambda step: step % args_cli.video_interval == 0,
            'video_length': args_cli.video_length,
            'disable_logger': True,
        }
        print('[INFO] Recording videos during training.')
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 包装为 RSL-RL 向量环境
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建 RSL-RL runner
    if agent_cfg.class_name == 'OnPolicyRunner':
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == 'DistillationRunner':
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == 'OnPolicyRunnerCTS':
        runner = OnPolicyRunnerCTS(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f'Unsupported runner class: {agent_cfg.class_name}')
    # 记录 git 状态到日志
    runner.add_git_repo_to_log(__file__)
    # 加载检查点
    if agent_cfg.resume or agent_cfg.algorithm.class_name == 'Distillation':
        print(f'[INFO]: Loading model checkpoint from: {resume_path}')
        runner.load(resume_path)

    # 将配置写入日志目录
    dump_yaml(os.path.join(log_dir, 'params', 'env.yaml'), env_cfg)
    dump_yaml(os.path.join(log_dir, 'params', 'agent.yaml'), agent_cfg)
    sys.stdout = Logger(os.path.join(log_dir, 'train.log'))
    # 运行训练
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 关闭仿真器
    env.close()


if __name__ == '__main__':
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
