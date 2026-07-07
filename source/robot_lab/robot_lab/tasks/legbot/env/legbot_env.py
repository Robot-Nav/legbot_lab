from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2


class LegbotEnv(ManagerBasedRLEnv):
    """LegBot environment.

    Reuses Go2's ActionManagerGo2 which maintains _prev_prev_action
    for second-order action smoothness reward computation.
    """

    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        super().load_managers()
        # override action manager (same as Go2)
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print("[LegbotEnv-INFO] Overriding action manager with ActionManagerGo2: ", self.action_manager)
