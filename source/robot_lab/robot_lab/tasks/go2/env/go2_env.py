"""Go2 强化学习环境定义。

包含标准环境以及支持动作延迟的环境变体。
"""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2, ActionManagerGo2WithDelay

import torch


class Go2Env(ManagerBasedRLEnv):
    """标准 Go2 强化学习环境。"""

    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        """加载管理器并使用自定义动作管理器覆盖默认实现。"""
        super().load_managers()
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print('[Go2Env-INFO] 使用 ActionManagerGo2 覆盖动作管理器: ', self.action_manager)


class ActionDelayGo2Env(ManagerBasedRLEnv):
    """支持动作延迟的 Go2 强化学习环境。"""

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化动作延迟环境。

        参数:
            cfg: 环境配置。
            render_mode: 渲染模式，例如 ``human`` 或 ``rgb_array``；默认为 ``None``。
            **kwargs: 其他自定义关键字参数。
        """
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        print(
            '[ActionDelayGo2Env-WARNING] 正在使用 ActionDelayGo2Env；'
            '请确保所有 ActionTerm 都支持在单个 step() 内被多次调用 process_actions()。'
        )

    def load_managers(self):
        """加载管理器并使用支持延迟的动作管理器覆盖默认实现。"""
        super().load_managers()
        self.action_manager = ActionManagerGo2WithDelay(self.cfg.actions, self)
        print('[ActionDelayGo2Env-INFO] 使用 ActionManagerGo2WithDelay 覆盖动作管理器: ', self.action_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """推进环境一个策略步。

        在 ``decimation`` 个物理步中根据随机延迟掩码应用动作，实现动作延迟。
        所有动作项的 ``process_actions`` 均需支持在单个 ``step()`` 内被多次调用。

        参数:
            action: 策略输出的动作，形状为 ``(num_envs, action_dim)``。

        返回:
            观测、奖励、终止标志、超时标志和额外信息的元组。
        """
        self.action_manager.update_action(action.to(self.device))

        actions_start_decimation = torch.randint(0, self.cfg.decimation + 1, (self.num_envs, 1), device=self.device)

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1

            action_delay_masks = i < actions_start_decimation
            self.action_manager.process_action_with_delay(action_delay_masks)

            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if 'interval' in self.event_manager.available_modes:
            self.event_manager.apply(mode='interval', dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
