"""Go2 自定义动作管理器。

维护上上步动作以计算二阶动作平滑性奖励，并提供支持动作延迟的变体。
"""

from isaaclab.managers import ActionManager
import torch
from collections.abc import Sequence


class ActionManagerGo2(ActionManager):
    """基础 Go2 动作管理器，保留 ``_prev_prev_action`` 用于动作平滑性奖励。"""

    def __init__(self, *args, **kwargs):
        """初始化并分配上上步动作缓存。"""
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """重置指定环境的动作历史。

        参数:
            env_ids: 待重置的环境索引；若为 ``None`` 则重置全部。

        返回:
            空字典，与父类接口保持一致。
        """
        super().reset(env_ids)
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0
        return {}

    def process_action(self, action: torch.Tensor):
        """处理发送给环境的动作。

        注意:
            本函数每环境步调用一次。

        参数:
            action: 要处理的动作。
        """
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f'动作形状不合法，期望: {self.total_action_dim}，实际: {action.shape[1]}。')
        # 更新动作历史
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # 将动作分发到各动作项
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    @property
    def prev_prev_action(self):
        """返回上上步动作。"""
        return self._prev_prev_action


class ActionManagerGo2WithDelay(ActionManager):
    """支持动作延迟的 Go2 动作管理器。

    通过 ``process_action_with_delay`` 在单步内多次应用动作，实现随机动作延迟。
    """

    def __init__(self, *args, **kwargs):
        """初始化并分配上上步动作缓存。"""
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """重置指定环境的动作历史。

        参数:
            env_ids: 待重置的环境索引；若为 ``None`` 则重置全部。

        返回:
            空字典，与父类接口保持一致。
        """
        super().reset(env_ids)
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0
        return {}

    def process_action(self, action: torch.Tensor):
        """本管理器不使用标准 ``process_action``，请使用 ``update_action`` 与 ``process_action_with_delay``。"""
        raise NotImplementedError('对于 ActionManagerWithDelay，请使用 update_action() 与 process_action_with_delay()。')

    def process_action_with_delay(self, action_delay_masks: torch.Tensor):
        """根据延迟掩码处理动作。

        重要说明:
            本函数可在单个 ``step()`` 内被多次调用，以实现动作延迟。
            所有动作项的 ``process_actions`` 都必须支持被多次调用。

        参数:
            action_delay_masks: 形状为 ``(num_envs, 1)`` 的布尔张量，为 ``True`` 时延用上一动作。
        """
        action = torch.where(action_delay_masks, self._prev_action, self._action)

        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def update_action(self, action: torch.Tensor):
        """更新当前动作并滚动动作历史。

        参数:
            action: 策略输出的当前动作。
        """
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f'动作形状不合法，期望: {self.total_action_dim}，实际: {action.shape[1]}。')
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

    @property
    def prev_prev_action(self):
        """返回上上步动作。"""
        return self._prev_prev_action
