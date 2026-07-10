# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""通用工具函数：激活函数/优化器解析、轨迹处理、可调用对象解析等。"""

from __future__ import annotations

import importlib
import pkgutil
import torch
import warnings
from tensordict import TensorDict
from typing import Any, Callable

import rsl_rl


def get_param(param: Any, idx: int) -> Any:
    """获取索引对应的参数值。

    参数:
        param: 单个参数或参数列表/元组。
        idx: 目标索引。
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """根据名称解析激活函数。

    参数:
        act_name: 激活函数名称。

    返回:
        对应的激活函数模块。

    抛出:
        ValueError: 未找到对应激活函数时。
    """
    act_dict = {
        'elu': torch.nn.ELU(),
        'selu': torch.nn.SELU(),
        'relu': torch.nn.ReLU(),
        'crelu': torch.nn.CELU(),
        'lrelu': torch.nn.LeakyReLU(),
        'tanh': torch.nn.Tanh(),
        'sigmoid': torch.nn.Sigmoid(),
        'softplus': torch.nn.Softplus(),
        'gelu': torch.nn.GELU(),
        'swish': torch.nn.SiLU(),
        'mish': torch.nn.Mish(),
        'identity': torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f'无效激活函数：{act_name}，可用：{list(act_dict.keys())}')


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """根据名称解析优化器。

    参数:
        optimizer_name: 优化器名称。

    返回:
        对应的优化器类。

    抛出:
        ValueError: 未找到对应优化器时。
    """
    optimizer_dict = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f'无效优化器：{optimizer_name}，可用：{list(optimizer_dict.keys())}')


def split_and_pad_trajectories(
    tensor: torch.Tensor | TensorDict, dones: torch.Tensor
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
    """在终止位置切分轨迹并填充到相同长度。

    将轨迹按终止标志切分、拼接，并用零填充至最长轨迹长度；
    同时返回标识有效部分的掩码。

    输入维度顺序：[time, num_envs, ...]

    示例：
        输入：[[a1, a2, a3, a4 | a5, a6],
              [b1, b2 | b3, b4, b5 | b6]]

        输出：[[a1, a2, a3, a4],  | [[True, True, True, True],
              [a5, a6, 0, 0],    |  [True, True, False, False],
              [b1, b2, 0, 0],    |  [True, True, False, False],
              [b3, b4, b5, 0],   |  [True, True, True, False],
              [b6, 0, 0, 0]]     |  [True, False, False, False]]
    """
    dones = dones.clone()
    dones[-1] = 1
    # 调整维度顺序为 (num_envs, time, ...) 以便正确 reshape
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # 通过统计连续未终止元素数量得到轨迹长度
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # 提取每条轨迹
    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # 切分轨迹
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # 添加一条完整长度的零轨迹以支持 pad_sequence
            trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], device=v.device))
            # 填充至最长轨迹
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
            # 移除添加的零轨迹
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)], device=tensor.device
        )
    else:
        # 切分轨迹
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # 添加一条完整长度的零轨迹以支持 pad_sequence
        trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
        # 填充至最长轨迹
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)  # type: ignore
        # 移除添加的零轨迹
        padded_trajectories = padded_trajectories[:, :-1]
    # 构造有效部分掩码
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories: torch.Tensor | TensorDict, masks: torch.Tensor) -> torch.Tensor | TensorDict:
    """split_and_pad_trajectories 的逆操作，将填充轨迹还原。"""
    # 通过转置与掩码还原原始形状
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def resolve_callable(callable_or_name: type | Callable | str) -> Callable:
    """将字符串、类型或可调用对象解析为可调用对象。

    支持以下格式：
        - 直接传入类型或函数（如 MyClass、my_func）。
        - 冒号分隔限定名：'module.path:Attr.Nested'（推荐）。
        - 点分隔限定名：'module.path.ClassName'。
        - 简单名称：如 'PPO'、'ActorCritic'（在 rsl_rl 包中查找）。

    参数:
        callable_or_name: 可调用对象或字符串名称。

    返回:
        解析后的可调用对象。

    抛出:
        TypeError: 输入既不是可调用对象也不是字符串。
        ImportError: 模块无法导入。
        AttributeError: 模块中找不到对应属性。
        ValueError: 简单名称在 rsl_rl 包中找不到。
    """
    # 直接传入可调用对象
    if callable(callable_or_name):
        return callable_or_name

    # 必须是字符串
    if not isinstance(callable_or_name, str):
        raise TypeError(f'期望可调用对象或字符串，得到 {type(callable_or_name)}')

    # 冒号分隔（如 'module.path:Attr.Nested'）
    if ':' in callable_or_name:
        module_path, attr_path = callable_or_name.rsplit(':', 1)
        module = importlib.import_module(module_path)
        obj = module
        for attr in attr_path.split('.'):
            obj = getattr(obj, attr)
        return obj  # type: ignore

    # 点分隔（如 'module.path.ClassName'）
    if '.' in callable_or_name:
        parts = callable_or_name.split('.')
        module_found = False
        for i in range(len(parts) - 1, 0, -1):
            module_path = '.'.join(parts[:i])
            attr_parts = parts[i:]
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                continue
            module_found = True
            obj = module
            try:
                for attr in attr_parts:
                    obj = getattr(obj, attr)
                return obj  # type: ignore
            except AttributeError:
                continue
        if module_found:
            raise AttributeError(f"无法解析 '{callable_or_name}'：模块中未找到对应属性")
        else:
            raise ImportError(f"无法解析 '{callable_or_name}'：未找到有效的 module.attr 切分")

    # 简单名称：在 rsl_rl 包中查找
    for _, module_name, _ in pkgutil.iter_modules(rsl_rl.__path__, 'rsl_rl.'):
        module = importlib.import_module(module_name)
        if hasattr(module, callable_or_name):
            return getattr(module, callable_or_name)

    # 全部失败则抛出异常
    raise ValueError(
        f"无法解析 '{callable_or_name}'。请使用 'module.path:ClassName' 形式的限定名或直接传入类。"
    )


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """校验观测配置并解析缺失的观测集合。

    输入 obs 包含环境返回的观测组，obs_groups 定义各观测集合使用的观测组列表，例如：
        {
            'policy': ['group_1', 'group_2'],
            'critic': ['group_1', 'group_3']
        }

    函数会检查 obs_groups 中所有观测组都存在于环境观测中。

    若 default_sets 中某项（如 'critic'）未在 obs_groups 中提供，则按以下规则默认填充：
        1. 若环境观测中存在同名组，则将该组作为该观测集合。
        2. 否则使用 'policy' 观测集合的观测组。

    参数:
        obs: 环境返回的观测字典。
        obs_groups: 观测集合配置。
        default_sets: 算法需要的保留观测集合名（除 'policy' 外）。

    返回:
        解析后的观测集合配置。

    抛出:
        ValueError: 观测集合为空列表，或包含环境不存在的观测组。
    """
    # 检查 policy 观测集合
    if 'policy' not in obs_groups:
        if 'policy' in obs:
            obs_groups['policy'] = ['policy']
            warnings.warn(
                "观测配置字典 'obs_groups' 必须包含 'policy' 键。"
                "由于环境观测中存在名为 'policy' 的组，已将其作为 policy 观测集合。"
                "建议显式在 'obs_groups' 中添加 'policy' 键。该默认行为将在未来版本移除。"
            )
        else:
            raise ValueError(
                f"观测配置字典 'obs_groups' 必须包含 'policy' 键。当前键：{list(obs_groups.keys())}"
            )

    # 校验所有观测集合
    for set_name, groups in obs_groups.items():
        # 不能为空列表
        if len(groups) == 0:
            msg = f"'obs_groups' 中的 '{set_name}' 键不能是空列表。"
            if set_name in default_sets:
                if set_name not in obs:
                    msg += "建议删除该键以默认使用 'policy' 集合的观测组。"
                else:
                    msg += f"建议删除该键以默认使用环境中的 '{set_name}' 观测组。"
            raise ValueError(msg)
        # 检查观测组是否存在于环境观测中
        for group in groups:
            if group not in obs:
                raise ValueError(
                    f"观测集合 '{set_name}' 中的观测组 '{group}' 不存在于环境观测中。"
                    f"环境可用观测组：{list(obs.keys())}"
                )

    # 填充缺失的默认观测集合
    for default_set_name in default_sets:
        if default_set_name not in obs_groups:
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"观测配置字典 'obs_groups' 必须包含 '{default_set_name}' 键。"
                    f"由于环境观测中存在名为 '{default_set_name}' 的组，已将其作为该观测集合。"
                    f"建议显式在 'obs_groups' 中添加 '{default_set_name}' 键。该默认行为将在未来版本移除。"
                )
            else:
                obs_groups[default_set_name] = obs_groups['policy'].copy()
                warnings.warn(
                    f"观测配置字典 'obs_groups' 必须包含 '{default_set_name}' 键。"
                    f"由于缺少 '{default_set_name}' 配置，已使用 'policy' 集合的观测组。"
                    f"建议显式在 'obs_groups' 中添加 '{default_set_name}' 键。该默认行为将在未来版本移除。"
                )

    # 打印最终解析结果
    print('-' * 80)
    print('解析后的观测集合：')
    for set_name, groups in obs_groups.items():
        print('\t', set_name, ': ', groups)
    print('-' * 80)

    return obs_groups
