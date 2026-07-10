"""LegBot 任务的 RSL-RL 训练器配置。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """标准 PPO 训练器配置。"""

    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 500
    experiment_name = 'legbot_rough'

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule='adaptive',
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlMoeCtsActorCriticCfg(RslRlPpoActorCriticCfg):
    """MoE-CTS 策略网络配置。"""

    class_name = 'ActorCriticMoECTS'
    init_noise_std = 1.0
    expert_num = 8
    latent_dim = 32
    norm_type = 'l2norm'
    teacher_encoder_hidden_dims = [512, 256]
    student_encoder_hidden_dims = [512, 256, 256]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'
    actor_obs_normalization = False
    critic_obs_normalization = False


@configclass
class RslRlMoeCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """MoE-CTS 算法配置。"""

    class_name = 'MoECTS'
    value_loss_coef = 1.0
    load_balance_coef = 0.01
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3
    student_encoder_learning_rate = 1e-3
    schedule = 'adaptive'
    gamma = 0.99
    lam = 0.95
    betas = (0.9, 0.999)
    weight_decay = 0.0
    desired_kl = 0.01
    max_grad_norm = 1.0
    teacher_env_ratio = 0.75


@configclass
class MoECTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    """MoE-CTS 训练器总配置。"""

    experiment_name = 'legbot_moe_cts'
    class_name = 'OnPolicyRunnerCTS'
    num_steps_per_env = 24
    max_iterations = 300000
    save_interval = 500
    empirical_normalization = False
    # 将观测集合映射到环境观测组：
    # policy 集合使用 policy 组（带历史），critic 集合使用 critic 组（特权观测）。
    # single_obs 组由 ActorCriticMoECTS 直接访问，不通过 obs_groups。
    obs_groups = {'policy': ['policy'], 'critic': ['critic']}
    policy = RslRlMoeCtsActorCriticCfg()
    algorithm = RslRlMoeCtsAlgorithmCfg()


# concat elu 设计受 concat relu 启发，参见 https://arxiv.org/pdf/2303.07507
@configclass
class MoECTSCatELURunnerCfg(MoECTSRunnerCfg):
    """使用 cat_elu 激活函数的 MoE-CTS 训练器配置。"""

    def __post_init__(self):
        """将策略激活函数改为 cat_elu。"""
        super().__post_init__()
        self.policy.activation = 'cat_elu'
