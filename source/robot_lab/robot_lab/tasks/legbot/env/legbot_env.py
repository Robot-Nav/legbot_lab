from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2


class LegbotEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        super().load_managers()
        # 复用 Go2 的自定义 ActionManager（维护 prev_prev_action，用于动作平滑奖励）
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print("[LegbotEnv-INFO] Overriding action manager with ActionManagerGo2: ", self.action_manager)
