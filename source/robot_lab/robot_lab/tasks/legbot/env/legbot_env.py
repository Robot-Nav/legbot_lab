"""LegBot 强化学习环境定义。

复用 Go2 的自定义动作管理器以支持动作平滑性奖励。
"""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2


class LegbotEnv(ManagerBasedRLEnv):
    """LegBot 四足机器人强化学习环境。"""

    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        """加载管理器并覆盖为 Go2 动作管理器。"""
        super().load_managers()
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print('[LegbotEnv-INFO] 使用 ActionManagerGo2 覆盖动作管理器: ', self.action_manager)
