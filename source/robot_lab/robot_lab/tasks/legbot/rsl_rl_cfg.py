from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlMoeCtsActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCriticMoECTS"
    init_noise_std = 1.0
    expert_num = 8
    latent_dim = 32
    norm_type = "l2norm"
    teacher_encoder_hidden_dims = [512, 256]
    student_encoder_hidden_dims = [512, 256, 256]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class RslRlMoeCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "MoECTS"
    value_loss_coef = 1.0
    load_balance_coef = 0.01
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    student_encoder_learning_rate = 1e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    betas = (0.9, 0.999)
    weight_decay = 0.0
    desired_kl = 0.01
    max_grad_norm = 1.0
    teacher_env_ratio = 0.75


@configclass
class LegbotMoECTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "legbot_wf_moe_cts"
    class_name = "OnPolicyRunnerCTS"
    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 500
    policy = RslRlMoeCtsActorCriticCfg()
    algorithm = RslRlMoeCtsAlgorithmCfg()
    obs_groups: dict = {"policy": ["policy"], "critic": ["critic"]}
    empirical_normalization = False
