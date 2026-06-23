from copy import deepcopy

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@configclass
class MoECTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    history_length = 5

    class_name = "OnPolicyRunnerCTS"

    policy = {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "teacher_encoder_hidden_dims": [512, 256],
        "student_encoder_hidden_dims": [512, 256, 256],
        "expert_num": 8,
        "activation": "elu",
        "latent_dim": 32,
        "norm_type": "l2norm",
    }

    algorithm = {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "student_encoder_learning_rate": 1e-3,
        "schedule": "adaptive",
        "gamma": 0.998,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "teacher_env_ratio": 0.75,
        "load_balance_coef": 0.01,
    }

    def to_dict(self):
        return {
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "empirical_normalization": self.empirical_normalization,
            "save_interval": self.save_interval,
            "history_length": self.history_length,
            "policy": deepcopy(self.policy),
            "algorithm": deepcopy(self.algorithm),
            "run_name": self.run_name,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
            "experiment_name": self.experiment_name,
        }
