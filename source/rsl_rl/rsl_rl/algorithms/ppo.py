# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO 算法实现（https://arxiv.org/abs/1707.06347）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable


class PPO:
    """近端策略优化算法（https://arxiv.org/abs/1707.06347）。"""

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN
    """演员-评论家模块。"""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = 'adaptive',
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = 'cpu',
        # RND 参数
        rnd_cfg: dict | None = None,
        # 对称性参数
        symmetry_cfg: dict | None = None,
        # 分布式训练参数
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """初始化 PPO 算法。

        参数:
            policy: 演员-评论家策略。
            storage: Rollout 存储。
            num_learning_epochs: 每次数据更新轮数。
            num_mini_batches: 每次更新 mini-batch 数量。
            clip_param: PPO 裁剪参数。
            gamma: 折扣因子。
            lam: GAE lambda。
            value_loss_coef: 价值损失系数。
            entropy_coef: 熵奖励系数。
            learning_rate: 学习率。
            max_grad_norm: 梯度裁剪阈值。
            use_clipped_value_loss: 是否使用裁剪价值损失。
            schedule: 学习率调度策略。
            desired_kl: 目标 KL 散度。
            normalize_advantage_per_mini_batch: 是否逐 mini-batch 归一化优势。
            device: 计算设备。
            rnd_cfg: RND 配置，可选。
            symmetry_cfg: 对称性配置，可选。
            multi_gpu_cfg: 多 GPU 配置，可选。
        """
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # 多 GPU 参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg['global_rank']
            self.gpu_world_size = multi_gpu_cfg['world_size']
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND 组件
        if rnd_cfg:
            # 提取 PPO 中使用的参数
            rnd_lr = rnd_cfg.pop('learning_rate', 1e-3)
            # 创建 RND 模块
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # 创建 RND 优化器
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # 对称性组件
        if symmetry_cfg is not None:
            # 检查是否启用对称性
            use_symmetry = symmetry_cfg['use_data_augmentation'] or symmetry_cfg['use_mirror_loss']
            # 若未用于学习，则仅用于日志
            if not use_symmetry:
                print('对称性未用于学习，将仅用于日志。')
            # 解析数据增强函数（支持字符串名称或直接可调用对象）
            symmetry_cfg['data_augmentation_func'] = resolve_callable(symmetry_cfg['data_augmentation_func'])
            # 检查配置有效性
            if not callable(symmetry_cfg['data_augmentation_func']):
                raise ValueError(
                    f'存在对称性配置，但函数不可调用：'
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # 检查策略是否兼容对称性
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError('对称性增强不支持循环策略。')
            # 存储对称性配置
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO 组件
        self.policy = policy
        self.policy.to(self.device)

        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 添加存储
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO 参数
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        """根据观测生成动作并记录 transition 信息。"""
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # 计算动作与价值
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # 在 env.step() 前记录观测
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """处理环境单步结果：更新归一化、记录奖励、处理超时自举并存储转移。"""
        # 更新归一化器
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # 记录奖励与终止标志
        # 注意：此处克隆奖励，后续会根据超时进行自举
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 计算内在奖励并叠加到外在奖励
        if self.rnd:
            # 计算内在奖励
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # 将内在奖励加到外在奖励
            self.transition.rewards += self.intrinsic_rewards

        # 对超时进行自举
        if 'time_outs' in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras['time_outs'].unsqueeze(1).to(self.device), 1
            )

        # 记录转移
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """使用 GAE 计算回报与优势。"""
        st = self.storage
        # 计算最后一步的价值
        last_values = self.policy.evaluate(obs).detach()
        # 计算回报与优势
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # 最后一步使用自举回报
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 非终止状态为 1，终止状态为 0
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD 误差：r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # 优势：A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # 回报：R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # 计算优势
        st.advantages = st.returns - st.values
        # 若未使用每 mini-batch 优势归一化，则全局归一化
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """执行一次策略更新，返回各损失均值。"""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND 损失
        mean_rnd_loss = 0 if self.rnd else None
        # 对称性损失
        mean_symmetry_loss = 0 if self.symmetry else None

        # 获取 mini-batch 生成器
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 遍历 batch
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # 每个样本的增强次数，无增强时为 1
            original_batch_size = obs_batch.batch_size[0]

            # 是否在每个 mini-batch 内归一化优势
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 执行对称增强
            if self.symmetry and self.symmetry['use_data_augmentation']:
                # 使用对称性进行数据增强
                data_augmentation_func = self.symmetry['data_augmentation_func']
                # 返回形状：[batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry['_env'],
                )
                # 计算每个样本的增强次数
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # 重复 batch 其余部分
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # 用当前策略重新计算动作对数概率与熵
            # 注意：策略参数已更新，因此需要重新计算
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # 仅保留第一次增强（原始样本）的均值、标准差与熵
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # 计算 KL 散度并自适应学习率
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # 在所有 GPU 间聚合 KL 散度
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # 仅在主进程更新学习率
                    # TODO: 是否需要？若各 GPU 的 KL 散度“相同”，则学习率也应相同。
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 同步所有 GPU 的学习率
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # 更新所有参数组的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # 替代损失
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 价值函数损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # 对称性损失
            if self.symmetry:
                # 获取对称动作
                # 注意：若之前已做增强，则无需再次增强
                if not self.symmetry['use_data_augmentation']:
                    data_augmentation_func = self.symmetry['data_augmentation_func']
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry['_env'])
                    # 计算每个样本的增强次数
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # 演员对对称增强观测预测的动作均值
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # 计算对称增强后的动作均值
                # 假设第一次增强为原始样本。不使用之前的 action_batch，因为那是从分布中采样的；
                # 而对称损失基于分布均值计算。
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry['_env']
                )

                # 计算损失
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # 将损失加入总损失
                if self.symmetry['use_mirror_loss']:
                    loss += self.symmetry['mirror_loss_coeff'] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND 损失
            # TODO: 将该处理移到 RND 模块内部。
            if self.rnd:
                # 提取 RND 状态
                # TODO: 检查是否仍需 torch.no_grad，目前仅为仿射变换。
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # 预测嵌入与目标嵌入
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # 使用均方误差计算损失
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # 计算 PPO 梯度
            self.optimizer.zero_grad()
            loss.backward()
            # 计算 RND 梯度
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # 收集所有 GPU 的梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用 PPO 梯度
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # 应用 RND 梯度
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 累计损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND 损失
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # 对称性损失
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # 用更新次数平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # 清空存储
        self.storage.clear()

        # 构建损失字典
        loss_dict = {
            'value': mean_value_loss,
            'surrogate': mean_surrogate_loss,
            'entropy': mean_entropy,
        }
        if self.rnd:
            loss_dict['rnd'] = mean_rnd_loss
        if self.symmetry:
            loss_dict['symmetry'] = mean_symmetry_loss

        return loss_dict

    def broadcast_parameters(self) -> None:
        """将模型参数广播到所有 GPU。"""
        # 获取当前 GPU 的模型参数
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 在所有 GPU 上加载源 GPU 的模型参数
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """收集所有 GPU 的梯度并取平均。

        该函数在反向传播后调用，用于同步所有 GPU 的梯度。
        """
        # 创建张量存储梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # 在所有 GPU 间平均梯度
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 获取所有参数
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # 用平均后的梯度更新所有参数
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制回数据
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数的偏移
                offset += numel
