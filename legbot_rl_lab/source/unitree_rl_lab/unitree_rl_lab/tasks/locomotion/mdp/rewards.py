# 自定义 MDP 奖励函数：覆盖能量、姿态、脚部接触、步态、镜像等奖励项。
# 启用未来类型注解特性
from __future__ import annotations

# 导入张量计算库
import torch
# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 从数学工具库导入四元数逆旋转函数，兼容不同版本函数名
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    # 回退到旧版函数名
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
# 导入机器人关节和刚体类
from isaaclab.assets import Articulation, RigidObject
# 导入场景实体配置类
from isaaclab.managers import SceneEntityCfg
# 导入接触传感器类
from isaaclab.sensors import ContactSensor

# 类型检查时导入强化学习环境类
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
关节惩罚函数
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚机器人关节消耗的能量"""
    # 获取机器人关节对象
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取关节速度和施加的力矩
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # 计算能量消耗：速度与力矩乘积的绝对值之和
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当机器人处于静止指令时，奖励关节位置接近默认位置"""
    # 获取机器人关节对象
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节位置与默认位置的偏差
    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    # 获取速度指令的范数
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    # 仅在指令接近零时施加奖励
    return reward * (cmd_norm < 0.1)


"""
机器人基座奖励函数
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用L2平方核奖励智能体使其重力方向对齐目标重力向量"""
    # 获取机器人刚体对象
    asset: RigidObject = env.scene[asset_cfg.name]

    # 将目标重力向量转换为张量
    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    # 计算当前重力投影与目标重力的余弦距离
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)
    # 将[-1, 1]映射到[0, 1]
    normalized = 0.5 * cos_dist + 0.5
    # 返回平方值
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚基座Z轴线速度偏离向上方向"""
    # 获取机器人刚体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算重力投影在Z轴分量的平方偏差
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """惩罚关节位置偏离默认位置的偏差"""
    # 获取机器人关节对象
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算速度指令范数
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    # 计算机器人基座水平速度范数
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    # 计算关节位置偏差
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    # 当有速度指令或基座运动时使用正常惩罚，否则使用缩放惩罚
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def hip_pos_penalty_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    command_threshold: float,
) -> torch.Tensor:
    """惩罚 hip 关节位置偏离默认位置（L1 范数），防止外八"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, [1, 2]]
    cmd_large = torch.any(torch.abs(command) > command_threshold, dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
        ord=1,
    )
    return torch.where(cmd_large, running_reward, stand_still_scale * running_reward)


"""
脚部接触奖励函数
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚脚部撞击垂直表面"""
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取脚部Z轴接触力
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    # 获取脚部XY平面接触力
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # 当水平力大于4倍垂直力时判定为撞击
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """奖励摆动脚在离地指定高度上的表现"""
    # 获取机器人刚体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算脚部相对于基座的位置（在世界坐标系中）
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    # 初始化脚部在基座坐标系中的位置张量
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    # 计算脚部相对于基座的速度
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    # 初始化脚部在基座坐标系中的速度张量
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    # 将所有脚部位置和速度转换到基座坐标系
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    # 计算脚部Z轴高度与目标的平方误差
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    # 使用tanh缩放水平速度
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    # 计算奖励：高度误差乘以速度缩放
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # 仅在速度指令非零时生效
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # 根据基座倾斜程度缩放奖励
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """奖励摆动脚在指定离地高度上的表现"""
    # 获取机器人刚体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    # 计算脚部在世界坐标系中的高度与目标高度的平方误差
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # 使用tanh缩放水平速度
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    # 计算加权奖励
    reward = foot_z_target_error * foot_velocity_tanh
    # 使用指数函数归一化奖励
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚脚部间距过近"""
    # 获取机器人关节对象
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取脚部位置
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # 计算两只脚之间的距离
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    # 当距离小于阈值时施加惩罚
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """当速度指令为零时，奖励脚部接触地面"""
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 判断脚部是否接触地面
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 获取速度指令范数
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    # 计算接触脚的数量作为奖励
    reward = torch.sum(is_contact, dim=-1).float()
    # 仅在指令接近零时施加奖励
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚各脚在空中和地面时间的方差"""
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 检查是否启用了空中时间跟踪
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # 获取各脚的空中时间和接触时间
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    # 计算空中时间和接触时间的方差之和（截断最大值0.5）
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
步态奖励函数
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """根据步态相位奖励脚部接触模式"""
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 判断脚部是否接触地面
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 计算全局相位（基于回合时间）
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    # 计算各腿的相位（考虑偏移）
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    # 初始化奖励张量
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    # 遍历每条腿，检查接触状态是否符合步态相位
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    # 如果指定了指令名称，仅在速度指令非零时施加奖励
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
其他奖励函数
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """奖励镜像关节位置的一致性"""
    # 获取机器人关节对象
    asset: Articulation = env.scene[asset_cfg.name]
    # 检查缓存中是否已保存镜像关节索引
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # 缓存所有镜像关节对的索引
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    # 初始化奖励张量
    reward = torch.zeros(env.num_envs, device=env.device)
    # 遍历所有镜像关节对
    for joint_pair in env.joint_mirror_joints_cache:
        # 计算关节位置偏差并累加
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    # 对镜像关节数量进行归一化
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward