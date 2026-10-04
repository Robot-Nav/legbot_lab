from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from robot_lab.tasks.go2.manager.action_manager import ActionManagerLegbot
from .observations import LegbotObservationManager


class LegbotEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        super().load_managers()
        # Policy actions and last_action follow motor order: three leg joints then one wheel per leg.
        self.action_manager = ActionManagerLegbot(self.cfg.actions, self)
        self.observation_manager = LegbotObservationManager(self.cfg.observations, self)
        print("[LegbotEnv-INFO] Using deployment-aligned CTS observations.")
        print("[LegbotEnv-INFO] Overriding action manager with ActionManagerLegbot: ", self.action_manager)
