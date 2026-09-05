import gymnasium as gym

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##
gym.register(
    id="RobotLab-Legbot-v0",
    entry_point="robot_lab.tasks.legbot.env.legbot_env:LegbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LegbotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:LegbotMoECTSRunnerCfg",
    },
)

# 防止导入子包中的配置
_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
