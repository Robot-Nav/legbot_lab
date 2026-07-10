# 基础版本来源：IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

import copy
import os
import torch
import re
import os
import sys

# 用于将终端输出记录到文件，并剥离 ANSI 转义码。
class Logger:
    """将标准输出同时写入终端与日志文件，并去除 ANSI 颜色码。"""

    def __init__(self, filename):
        """初始化日志记录器。

        Args:
            filename: 日志文件路径。
        """
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'w', encoding='utf-8')

        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        """写入终端与日志文件。

        Args:
            message: 需要输出的消息。
        """
        clean_message = self.ansi_escape.sub('', message)

        self.terminal.write(message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        """刷新终端与日志文件缓冲区。"""
        self.terminal.flush()
        self.log.flush()


def export_cts_policy_as_jit(policy: object, actor_obs_normalizer: object | None, single_obs_normalizer: object | None, path: str, filename='policy.pt'):
    """将 CTS 策略导出为 Torch JIT 文件。

    Args:
        policy: CTS 策略 torch 模块。
        actor_obs_normalizer: actor 观测的经验归一化模块；若为 None 则使用 Identity。
        single_obs_normalizer: 单帧观测的经验归一化模块；若为 None 则使用 Identity。
        path: 保存目录路径。
        filename: 导出的 JIT 文件名，默认为 'policy.pt'。
    """
    policy_exporter = _TorchPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer)
    policy_exporter.export(path, filename)


def export_cts_policy_as_onnx(
    policy: object, path: str, actor_obs_normalizer: object | None = None, single_obs_normalizer: object | None = None, filename='policy.onnx', verbose=False
):
    """将 CTS 策略导出为 Torch ONNX 文件。

    Args:
        policy: CTS 策略 torch 模块。
        actor_obs_normalizer: actor 观测的经验归一化模块；若为 None 则使用 Identity。
        single_obs_normalizer: 单帧观测的经验归一化模块；若为 None 则使用 Identity。
        path: 保存目录路径。
        filename: 导出的 ONNX 文件名，默认为 'policy.onnx'。
        verbose: 是否打印模型摘要，默认为 False。
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer, verbose)
    policy_exporter.export(path, filename)


# 辅助类 - 私有。


class _TorchPolicyExporter(torch.nn.Module):
    """将 actor-critic 策略导出为 JIT 文件。"""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None):
        """初始化 CTS 策略的 TorchScript 导出器。

        导出模型仅消费当前单帧 `single_obs`，并在内部维护历史帧栈，
        以重建学生编码器所需的 actor 观测。

        Args:
            policy: 待导出的源 CTS 策略模块。
            actor_obs_normalizer: 应用于堆叠 actor 观测的归一化模块。
            single_obs_normalizer: 应用于当前单帧观测的归一化模块。
        """
        assert not policy.is_recurrent, 'CTS policy should not be recurrent'
        super().__init__()

        # 复制策略参数
        if hasattr(policy, 'actor'):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, 'student'):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError('Policy does not have an actor/student module.')
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.state_dependent_std = policy.state_dependent_std
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f'num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs}).'
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        # 与部署侧 push_obs_history 保持一致的逐项历史布局：
        # [ang_vel(3), gravity(3), cmd(3), joint_pos(A), joint_vel(A), last_action(A)]。
        self.feature_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                'Unsupported single_obs layout: expected 3+3+3+3*num_actions to match num_single_obs.'
            )
        self.register_buffer('obs_history', torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

        # 若存在则复制归一化器
        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def forward(self, single_obs: torch.Tensor):
        """根据当前单帧观测计算策略动作。

        导出器内部维护一个 FIFO 历史缓冲区，每次前向传播时先平移历史，
        再追加最新观测。

        Args:
            single_obs: 当前步观测张量，形状为 `[B, num_single_obs]`。

        Returns:
            策略动作张量。
        """
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(
                f'Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}.'
            )
        if single_obs.shape[0] != 1:
            raise ValueError('TorchScript CTS deployment currently supports batch size 1 only.')

        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(self.obs_history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        else:
            return self.actor(latent_and_obs)

    @torch.jit.export
    def reset(self):
        """重置内部观测历史状态。"""
        self.obs_history.zero_()

    def export(self, path, filename):
        """导出模型到指定路径。

        Args:
            path: 保存目录。
            filename: 保存文件名。
        """
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """将 actor-critic 策略导出为 ONNX 文件。"""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None, verbose=False):
        """初始化 CTS 策略的 ONNX 导出器。

        Args:
            policy: 待导出的源 CTS 策略模块。
            actor_obs_normalizer: 应用于 actor 观测的归一化模块。
            single_obs_normalizer: 应用于单帧观测的归一化模块。
            verbose: 是否打印模型摘要。
        """
        assert not policy.is_recurrent, 'CTS policy should not be recurrent'
        super().__init__()
        self.verbose = verbose

        # 复制策略参数
        if hasattr(policy, 'actor'):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, 'student'):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError('Policy does not have an actor/student module.')
        self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
        self.num_actions = int(policy.num_actions)
        self.num_single_obs = int(policy.num_single_obs)
        self.num_actor_obs = int(policy.num_actor_obs)
        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f'num_actor_obs ({self.num_actor_obs}) must be divisible by num_single_obs ({self.num_single_obs}).'
            )
        self.history_len = self.num_actor_obs // self.num_single_obs
        # 与部署侧 push_obs_history 保持一致的逐项历史布局：
        # [ang_vel(3), gravity(3), cmd(3), joint_pos(A), joint_vel(A), last_action(A)]。
        self.feature_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        if sum(self.feature_dims) != self.num_single_obs:
            raise ValueError(
                'Unsupported single_obs layout: expected 3+3+3+3*num_actions to match num_single_obs.'
            )
        self.state_dependent_std = policy.state_dependent_std

        # 若存在则复制归一化器
        if actor_obs_normalizer:
            self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if single_obs_normalizer:
            self.single_obs_normalizer = copy.deepcopy(single_obs_normalizer)
        else:
            self.single_obs_normalizer = torch.nn.Identity()

    def _extract_single_obs_from_history(self, history: torch.Tensor) -> torch.Tensor:
        """从历史观测中提取最新单帧观测。

        Args:
            history: 堆叠的历史观测张量。

        Returns:
            单帧观测张量。
        """
        if history.dim() == 1:
            history = history.unsqueeze(0)
        if history.shape[-1] != self.num_actor_obs:
            raise ValueError(
                f'Expected history last dimension {self.num_actor_obs}, got {history.shape[-1]}.'
            )

        single_obs_terms = []
        offset = 0
        for dim in self.feature_dims:
            end = offset + dim * self.history_len
            single_obs_terms.append(history[:, end - dim:end])
            offset = end
        return torch.cat(single_obs_terms, dim=-1)

    def forward(self, history):
        """根据历史观测计算策略动作。

        Args:
            history: 堆叠的历史观测张量。

        Returns:
            策略动作张量。
        """
        if history.dim() == 1:
            history = history.unsqueeze(0)
        single_obs = self._extract_single_obs_from_history(history)
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        else:
            return self.actor(latent_and_obs)

    def export(self, path, filename):
        """导出模型到指定路径。

        Args:
            path: 保存目录。
            filename: 保存文件名。
        """
        self.to('cpu')
        self.eval()
        # 原使用 opset 11，但在 linux-aarch 上出现问题；18 在各平台表现良好。
        opset_version = 18
        torch.onnx.export(
            self,
            torch.zeros(1, self.num_actor_obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=['obs'],
            output_names=['actions'],
            dynamic_axes={},
        )
