"""
Unit tests for MoE+CTS modules.
Covers: modules.py, actor_critic_moe_cts.py, rollout_storage_cts.py, moe_cts.py
And an integration test simulating a full training loop.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import pytest
import torch
import torch.nn as nn

from unitree_rl_lab.rl.modules import (
    MLP,
    L2Norm,
    SimNorm,
    Experts,
    MoE,
    StudentMoEEncoder,
    get_activation,
)
from unitree_rl_lab.rl.actor_critic_moe_cts import ActorCriticMoECTS
from unitree_rl_lab.rl.rollout_storage_cts import RolloutStorageCTS
from unitree_rl_lab.rl.moe_cts import MoECTS


# ============================================================
# Fixtures
# ============================================================

NUM_OBS = 45
NUM_PRIV_OBS = 263  # 247 + foot_contact_forces(4) + joint_acc(12)
NUM_ACTIONS = 12
NUM_ENVS = 16
HISTORY_LENGTH = 5
EXPERT_NUM = 4
LATENT_DIM = 32


@pytest.fixture
def model():
    return ActorCriticMoECTS(
        num_actor_obs=NUM_OBS,
        num_critic_obs=NUM_PRIV_OBS,
        num_actions=NUM_ACTIONS,
        num_envs=NUM_ENVS,
        history_length=HISTORY_LENGTH,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        teacher_encoder_hidden_dims=[64],
        student_encoder_hidden_dims=[64, 32],
        expert_num=EXPERT_NUM,
        activation="elu",
        init_noise_std=1.0,
        latent_dim=LATENT_DIM,
        norm_type="l2norm",
    )


@pytest.fixture
def alg(model):
    return MoECTS(
        model=model,
        num_envs=NUM_ENVS,
        history_length=HISTORY_LENGTH,
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        load_balance_coef=0.01,
        learning_rate=1e-3,
        student_encoder_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        teacher_env_ratio=0.75,
        device="cpu",
    )


# ============================================================
# Test: modules.py
# ============================================================


class TestGetActivation:
    def test_valid_activations(self):
        for name in ["elu", "selu", "relu", "crelu", "lrelu", "tanh", "sigmoid"]:
            act = get_activation(name)
            assert act is not None, f"Activation '{name}' returned None"

    def test_crelu_is_relu(self):
        assert isinstance(get_activation("crelu"), nn.ReLU)

    def test_invalid_activation(self):
        act = get_activation("nonexistent")
        assert act is None


class TestL2Norm:
    def test_output_on_unit_sphere(self):
        norm = L2Norm()
        x = torch.randn(32, 64)
        y = norm(x)
        norms = torch.norm(y, p=2.0, dim=-1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-6)

    def test_zero_input(self):
        norm = L2Norm()
        x = torch.zeros(4, 16)
        y = norm(x)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

    def test_gradient_flows(self):
        norm = L2Norm()
        x = torch.randn(4, 16, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestSimNorm:
    def test_output_sums_to_one(self):
        norm = SimNorm()
        x = torch.randn(4, 32)
        y = norm(x)
        chunk_sums = y.view(4, -1, 8).sum(dim=-1)
        assert torch.allclose(chunk_sums, torch.ones(4, 4), atol=1e-6)

    def test_latent_dim_not_divisible_by_8(self):
        norm = SimNorm()
        x = torch.randn(4, 30)
        with pytest.raises(RuntimeError):
            norm(x)


class TestMLP:
    def test_output_shape(self):
        mlp = MLP([16, 32, 8], activation="elu")
        x = torch.randn(4, 16)
        y = mlp(x)
        assert y.shape == (4, 8)

    def test_last_activation(self):
        mlp = MLP([16, 32, 8], activation="relu", last_activation=True)
        x = torch.randn(4, 16)
        y = mlp(x)
        assert (y >= 0).all()

    def test_single_layer(self):
        mlp = MLP([16, 8])
        x = torch.randn(4, 16)
        y = mlp(x)
        assert y.shape == (4, 8)


class TestExperts:
    def test_output_shape(self):
        experts = Experts(
            expert_num=4, input_dim=32, backbone_hidden_dims=[64],
            expert_hidden_dim=32, output_dim=8,
        )
        x = torch.randn(8, 32)
        out = experts(x)
        assert out.shape == (8, 4, 8)

    def test_experts_produce_different_outputs(self):
        experts = Experts(
            expert_num=4, input_dim=32, backbone_hidden_dims=[64],
            expert_hidden_dim=32, output_dim=8,
        )
        x = torch.randn(8, 32)
        out = experts(x)
        for i in range(3):
            assert not torch.allclose(out[:, i], out[:, i + 1], atol=1e-6)


class TestMoE:
    def test_output_shape(self):
        moe = MoE(expert_num=4, input_dim=32, hidden_dims=[64, 32], output_dim=8)
        x = torch.randn(8, 32)
        output, weights = moe(x)
        assert output.shape == (8, 8)
        assert weights.shape == (8, 4)

    def test_gating_weights_sum_to_one(self):
        moe = MoE(expert_num=4, input_dim=32, hidden_dims=[64, 32], output_dim=8)
        x = torch.randn(8, 32)
        _, weights = moe(x)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(8), atol=1e-6)

    def test_gating_weights_non_negative(self):
        moe = MoE(expert_num=4, input_dim=32, hidden_dims=[64, 32], output_dim=8)
        x = torch.randn(8, 32)
        _, weights = moe(x)
        assert (weights >= 0).all()


class TestStudentMoEEncoder:
    def test_l2norm_output(self):
        enc = StudentMoEEncoder(
            expert_num=4, input_dim=32, hidden_dims=[64, 32],
            output_dim=16, norm_type="l2norm",
        )
        x = torch.randn(8, 32)
        latent, weights = enc(x)
        assert latent.shape == (8, 16)
        assert weights.shape == (8, 4)
        norms = torch.norm(latent, p=2.0, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-6)

    def test_simnorm_output(self):
        enc = StudentMoEEncoder(
            expert_num=4, input_dim=32, hidden_dims=[64, 32],
            output_dim=16, norm_type="simnorm",
        )
        x = torch.randn(8, 32)
        latent, weights = enc(x)
        assert latent.shape == (8, 16)


# ============================================================
# Test: actor_critic_moe_cts.py
# ============================================================


class TestActorCriticMoECTS:
    def test_init_shapes(self, model):
        assert model.history.shape == (NUM_ENVS, HISTORY_LENGTH, NUM_OBS)
        assert model.std.shape == (NUM_ACTIONS,)

    def test_teacher_act(self, model):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        actions = model.act(obs, priv, history, is_teacher=True)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_student_act(self, model):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        actions = model.act(obs, priv, history, is_teacher=False)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_teacher_evaluate(self, model):
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        value = model.evaluate(priv, history, is_teacher=True)
        assert value.shape == (NUM_ENVS, 1)

    def test_student_evaluate(self, model):
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        value = model.evaluate(priv, history, is_teacher=False)
        assert value.shape == (NUM_ENVS, 1)

    def test_act_inference(self, model):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        actions = model.act_inference(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_act_inference_history_updates(self, model):
        obs1 = torch.randn(NUM_ENVS, NUM_OBS)
        obs2 = torch.randn(NUM_ENVS, NUM_OBS)
        model.act_inference(obs1)
        history_after_1 = model.history.clone()
        model.act_inference(obs2)
        assert not torch.equal(model.history, history_after_1)
        assert torch.equal(model.history[:, -1, :], obs2)

    def test_reset_clears_history(self, model):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        model.act_inference(obs)
        dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        dones[0] = 1
        dones[3] = 1
        history_before = model.history.clone()
        model.reset(dones)
        assert torch.all(model.history[0] == 0.0)
        assert torch.all(model.history[3] == 0.0)
        assert torch.equal(model.history[1], history_before[1])
        assert torch.equal(model.history[2], history_before[2])

    def test_teacher_latent_on_unit_sphere(self, model):
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        latent = model.teacher_encoder(priv)
        norms = torch.norm(latent, p=2.0, dim=-1)
        assert torch.allclose(norms, torch.ones(NUM_ENVS), atol=1e-6)

    def test_student_latent_on_unit_sphere(self, model):
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        latent, _ = model.student_moe_encoder(history)
        norms = torch.norm(latent, p=2.0, dim=-1)
        assert torch.allclose(norms, torch.ones(NUM_ENVS), atol=1e-6)

    def test_evaluate_latent_detach(self, model):
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        value = model.evaluate(priv, history, is_teacher=True)
        value.sum().backward()
        assert model.teacher_encoder[0].network[0].weight.grad is None

    def test_simnorm_model(self):
        model_sim = ActorCriticMoECTS(
            num_actor_obs=NUM_OBS,
            num_critic_obs=NUM_PRIV_OBS,
            num_actions=NUM_ACTIONS,
            num_envs=NUM_ENVS,
            history_length=HISTORY_LENGTH,
            actor_hidden_dims=[64, 32],
            critic_hidden_dims=[64, 32],
            teacher_encoder_hidden_dims=[64],
            student_encoder_hidden_dims=[64, 32],
            expert_num=EXPERT_NUM,
            latent_dim=LATENT_DIM,
            norm_type="simnorm",
        )
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        latent = model_sim.teacher_encoder(priv)
        assert latent.shape == (NUM_ENVS, LATENT_DIM)

    def test_invalid_norm_type(self):
        with pytest.raises(AssertionError):
            ActorCriticMoECTS(
                num_actor_obs=NUM_OBS,
                num_critic_obs=NUM_PRIV_OBS,
                num_actions=NUM_ACTIONS,
                num_envs=NUM_ENVS,
                history_length=HISTORY_LENGTH,
                norm_type="invalid",
            )


# ============================================================
# Test: rollout_storage_cts.py
# ============================================================


class TestRolloutStorageCTS:
    def test_add_and_clear(self):
        storage = RolloutStorageCTS(
            num_envs=NUM_ENVS,
            teacher_num_envs=12,
            history_length=HISTORY_LENGTH,
            num_transitions_per_env=4,
            obs_shape=(NUM_OBS,),
            privileged_obs_shape=(NUM_PRIV_OBS,),
            actions_shape=(NUM_ACTIONS,),
        )
        trans = RolloutStorageCTS.Transition()
        trans.observations = torch.randn(NUM_ENVS, NUM_OBS)
        trans.critic_observations = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        trans.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.rewards = torch.randn(NUM_ENVS)
        trans.dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        trans.values = torch.randn(NUM_ENVS, 1)
        trans.actions_log_prob = torch.randn(NUM_ENVS)
        trans.action_mean = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.action_sigma = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

        for _ in range(4):
            storage.add_transitions(trans)
        assert storage.step == 4

        with pytest.raises(AssertionError):
            storage.add_transitions(trans)

        storage.clear()
        assert storage.step == 0

    def test_compute_returns(self):
        num_trans = 8
        storage = RolloutStorageCTS(
            num_envs=NUM_ENVS,
            teacher_num_envs=12,
            history_length=HISTORY_LENGTH,
            num_transitions_per_env=num_trans,
            obs_shape=(NUM_OBS,),
            privileged_obs_shape=(NUM_PRIV_OBS,),
            actions_shape=(NUM_ACTIONS,),
        )
        trans = RolloutStorageCTS.Transition()
        trans.observations = torch.randn(NUM_ENVS, NUM_OBS)
        trans.critic_observations = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        trans.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.rewards = torch.ones(NUM_ENVS) * 0.1
        trans.dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        trans.values = torch.randn(NUM_ENVS, 1)
        trans.actions_log_prob = torch.randn(NUM_ENVS)
        trans.action_mean = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.action_sigma = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

        for _ in range(num_trans):
            storage.add_transitions(trans)

        last_values = torch.randn(NUM_ENVS, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)
        assert storage.returns.shape == (num_trans, NUM_ENVS, 1)
        assert storage.advantages.shape == (num_trans, NUM_ENVS, 1)
        assert not torch.isnan(storage.returns).any()
        assert not torch.isnan(storage.advantages).any()

    def test_mini_batch_generator_teacher_student_split(self):
        num_trans = 4
        teacher_num = 12
        student_num = NUM_ENVS - teacher_num
        storage = RolloutStorageCTS(
            num_envs=NUM_ENVS,
            teacher_num_envs=teacher_num,
            history_length=HISTORY_LENGTH,
            num_transitions_per_env=num_trans,
            obs_shape=(NUM_OBS,),
            privileged_obs_shape=(NUM_PRIV_OBS,),
            actions_shape=(NUM_ACTIONS,),
        )
        trans = RolloutStorageCTS.Transition()
        trans.observations = torch.randn(NUM_ENVS, NUM_OBS)
        trans.critic_observations = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        trans.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.rewards = torch.randn(NUM_ENVS)
        trans.dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        trans.values = torch.randn(NUM_ENVS, 1)
        trans.actions_log_prob = torch.randn(NUM_ENVS)
        trans.action_mean = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.action_sigma = torch.randn(NUM_ENVS, NUM_ACTIONS)
        trans.history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

        for _ in range(num_trans):
            storage.add_transitions(trans)

        last_values = torch.randn(NUM_ENVS, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

        num_mini_batches = 2
        num_epochs = 1
        batches = list(storage.mini_batch_generator(num_mini_batches, num_epochs))
        assert len(batches) == num_mini_batches * num_epochs

        for batch in batches:
            obs_batch = batch[0]
            expected_batch = (teacher_num * num_trans // num_mini_batches
                              + student_num * num_trans // num_mini_batches)
            assert obs_batch.shape[0] == expected_batch

    def test_no_privileged_obs(self):
        storage = RolloutStorageCTS(
            num_envs=NUM_ENVS,
            teacher_num_envs=12,
            history_length=HISTORY_LENGTH,
            num_transitions_per_env=4,
            obs_shape=(NUM_OBS,),
            privileged_obs_shape=(None,),
            actions_shape=(NUM_ACTIONS,),
        )
        assert storage.privileged_observations is None


# ============================================================
# Test: moe_cts.py
# ============================================================


class TestMoECTS:
    def test_teacher_student_split(self, alg):
        assert alg.teacher_num_envs == 12
        assert alg.student_num_envs == 4
        assert len(alg.teacher_env_idxs) == 12
        assert len(alg.student_env_idxs) == 4
        assert len(set(alg.teacher_env_idxs.tolist()) | set(alg.student_env_idxs.tolist())) == NUM_ENVS

    def test_act_returns_correct_shape(self, alg):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        actions = alg.act(obs, priv, history)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_act_actions_are_real_valued(self, alg):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        actions = alg.act(obs, priv, history)
        assert not torch.isnan(actions).any()
        assert not torch.isinf(actions).any()

    def test_process_env_step(self, alg):
        alg.init_storage(NUM_ENVS, 4, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        actions = alg.act(obs, priv, history)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        infos = {}
        alg.process_env_step(rewards, dones, infos)
        assert alg.storage.step == 1

    def test_process_env_step_with_timeouts(self, alg):
        alg.init_storage(NUM_ENVS, 4, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        actions = alg.act(obs, priv, history)
        rewards = torch.ones(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        infos = {"time_outs": torch.ones(NUM_ENVS)}
        alg.process_env_step(rewards, dones, infos)
        assert alg.storage.step == 1

    def test_compute_returns(self, alg):
        num_trans = 4
        alg.init_storage(NUM_ENVS, num_trans, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        for _ in range(num_trans):
            actions = alg.act(obs, priv, history)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS, dtype=torch.long)
            alg.process_env_step(rewards, dones, {})
        alg.compute_returns(priv, history)
        assert not torch.isnan(alg.storage.returns).any()

    def test_update_returns_five_losses(self, alg):
        num_trans = 4
        alg.init_storage(NUM_ENVS, num_trans, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)
        for _ in range(num_trans):
            actions = alg.act(obs, priv, history)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS, dtype=torch.long)
            alg.process_env_step(rewards, dones, {})
        alg.compute_returns(priv, history)
        losses = alg.update()
        assert len(losses) == 5
        for loss in losses:
            assert isinstance(loss, float)
            assert not (loss != loss)

    def test_optimizer1_params_no_student(self, alg):
        student_params = set(id(p) for p in alg.model.student_moe_encoder.parameters())
        for group in alg.optimizer1.param_groups:
            for p in group["params"]:
                assert id(p) not in student_params, "Optimizer1 should not contain student encoder params"

    def test_optimizer2_params_only_student(self, alg):
        student_params = set(id(p) for p in alg.model.student_moe_encoder.parameters())
        opt2_params = set()
        for group in alg.optimizer2.param_groups:
            for p in group["params"]:
                opt2_params.add(id(p))
        assert student_params == opt2_params

    def test_load_balance_coef_effect(self, model):
        alg_low = MoECTS(
            model=model, num_envs=NUM_ENVS, history_length=HISTORY_LENGTH,
            load_balance_coef=0.0, device="cpu",
        )
        alg_high = MoECTS(
            model=model, num_envs=NUM_ENVS, history_length=HISTORY_LENGTH,
            load_balance_coef=1.0, device="cpu",
        )
        assert alg_low.load_balance_coef == 0.0
        assert alg_high.load_balance_coef == 1.0

    def test_small_num_envs(self):
        small_model = ActorCriticMoECTS(
            num_actor_obs=NUM_OBS,
            num_critic_obs=NUM_PRIV_OBS,
            num_actions=NUM_ACTIONS,
            num_envs=4,
            history_length=HISTORY_LENGTH,
            actor_hidden_dims=[64, 32],
            critic_hidden_dims=[64, 32],
            teacher_encoder_hidden_dims=[64],
            student_encoder_hidden_dims=[64, 32],
            expert_num=EXPERT_NUM,
            latent_dim=LATENT_DIM,
        )
        alg = MoECTS(
            model=small_model,
            num_envs=4,
            history_length=HISTORY_LENGTH,
            teacher_env_ratio=0.75,
            device="cpu",
        )
        assert alg.teacher_num_envs == 3
        assert alg.student_num_envs == 1

    def test_reset_dones_clears_history(self, alg):
        obs = torch.randn(NUM_ENVS, NUM_OBS)
        alg.model.act_inference(obs)
        dones = torch.zeros(NUM_ENVS, dtype=torch.long)
        dones[0] = 1
        alg.model.reset(dones)
        assert torch.all(alg.model.history[0] == 0.0)


# ============================================================
# Integration Test: Full Training Loop Simulation
# ============================================================


class TestIntegrationTrainingLoop:
    def test_full_training_iteration(self, alg):
        num_trans = 4
        alg.init_storage(NUM_ENVS, num_trans, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])

        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.zeros(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

        for step_i in range(num_trans):
            actions = alg.act(obs, priv, history)
            rewards = torch.randn(NUM_ENVS) * 0.1
            dones = torch.zeros(NUM_ENVS, dtype=torch.long)
            if step_i == num_trans - 1:
                dones[:] = 1
            infos = {"time_outs": dones.float()}
            alg.process_env_step(rewards, dones, infos)

        alg.compute_returns(priv, history)
        losses = alg.update()

        v_loss, s_loss, ent_loss, lat_loss, lb_loss = losses
        assert not (v_loss != v_loss), "Value loss is NaN"
        assert not (s_loss != s_loss), "Surrogate loss is NaN"
        assert not (lat_loss != lat_loss), "Latent loss is NaN"
        assert not (lb_loss != lb_loss), "Load balance loss is NaN"

    def test_multiple_iterations(self, alg):
        num_trans = 4
        alg.init_storage(NUM_ENVS, num_trans, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])

        for iteration in range(3):
            obs = torch.randn(NUM_ENVS, NUM_OBS)
            priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
            history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

            for step_i in range(num_trans):
                actions = alg.act(obs, priv, history)
                rewards = torch.randn(NUM_ENVS) * 0.1
                dones = torch.zeros(NUM_ENVS, dtype=torch.long)
                infos = {}
                alg.process_env_step(rewards, dones, infos)

            alg.compute_returns(priv, history)
            losses = alg.update()
            assert len(losses) == 5

    def test_latent_alignment_improves(self, model):
        alg = MoECTS(
            model=model,
            num_envs=NUM_ENVS,
            history_length=HISTORY_LENGTH,
            num_learning_epochs=4,
            num_mini_batches=2,
            learning_rate=1e-3,
            student_encoder_learning_rate=1e-3,
            load_balance_coef=0.01,
            device="cpu",
        )
        num_trans = 4
        alg.init_storage(NUM_ENVS, num_trans, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])

        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

        with torch.no_grad():
            teacher_latent = model.teacher_encoder(priv)
            student_latent, _ = model.student_moe_encoder(history)
            initial_dist = (teacher_latent - student_latent).pow(2).mean().item()

        for _ in range(10):
            obs = torch.randn(NUM_ENVS, NUM_OBS)
            for step_i in range(num_trans):
                actions = alg.act(obs, priv, history)
                rewards = torch.randn(NUM_ENVS) * 0.1
                dones = torch.zeros(NUM_ENVS, dtype=torch.long)
                alg.process_env_step(rewards, dones, {})
            alg.compute_returns(priv, history)
            alg.update()

        with torch.no_grad():
            student_latent_after, _ = model.student_moe_encoder(history)
            final_dist = (teacher_latent - student_latent_after).pow(2).mean().item()

        assert final_dist < initial_dist, (
            f"Distillation should reduce latent distance: "
            f"initial={initial_dist:.4f}, final={final_dist:.4f}"
        )

    def test_gating_diversity_across_speeds(self, model):
        alg = MoECTS(
            model=model,
            num_envs=NUM_ENVS,
            history_length=HISTORY_LENGTH,
            device="cpu",
        )
        obs_slow = torch.randn(NUM_ENVS, NUM_OBS)
        obs_slow[:, 6] = 0.2
        obs_slow[:, 7] = 0.0
        obs_slow[:, 8] = 0.0

        obs_fast = torch.randn(NUM_ENVS, NUM_OBS)
        obs_fast[:, 6] = 1.0
        obs_fast[:, 7] = 0.0
        obs_fast[:, 8] = 0.0

        obs_turn = torch.randn(NUM_ENVS, NUM_OBS)
        obs_turn[:, 6] = 0.5
        obs_turn[:, 7] = 0.0
        obs_turn[:, 8] = 1.0

        history_slow = obs_slow.unsqueeze(1).expand(-1, HISTORY_LENGTH, -1).flatten(1)
        history_fast = obs_fast.unsqueeze(1).expand(-1, HISTORY_LENGTH, -1).flatten(1)
        history_turn = obs_turn.unsqueeze(1).expand(-1, HISTORY_LENGTH, -1).flatten(1)

        with torch.no_grad():
            _, weights_slow = model.student_moe_encoder(history_slow)
            _, weights_fast = model.student_moe_encoder(history_fast)
            _, weights_turn = model.student_moe_encoder(history_turn)

        assert weights_slow.shape == (NUM_ENVS, EXPERT_NUM)
        assert weights_fast.shape == (NUM_ENVS, EXPERT_NUM)
        assert weights_turn.shape == (NUM_ENVS, EXPERT_NUM)

        mean_slow = weights_slow.mean(dim=0)
        mean_fast = weights_fast.mean(dim=0)
        mean_turn = weights_turn.mean(dim=0)

        assert not torch.allclose(mean_slow, mean_fast, atol=0.01) or \
               not torch.allclose(mean_slow, mean_turn, atol=0.01), \
               "Gating weights should differ across speed commands"

    def test_history_sliding_window(self, model):
        model.eval()
        torch.manual_seed(42)
        obs_list = []
        for t in range(HISTORY_LENGTH + 2):
            obs = torch.randn(NUM_ENVS, NUM_OBS)
            obs_list.append(obs)
            model.act_inference(obs)

        for t in range(HISTORY_LENGTH):
            expected_obs = obs_list[t + 2]
            assert torch.allclose(model.history[:, t, :], expected_obs, atol=1e-6)

    def test_partial_dones_during_rollout(self, alg):
        num_trans = 8
        alg.init_storage(NUM_ENVS, num_trans, [NUM_OBS], [NUM_PRIV_OBS], [NUM_ACTIONS])

        obs = torch.randn(NUM_ENVS, NUM_OBS)
        priv = torch.randn(NUM_ENVS, NUM_PRIV_OBS)
        history = torch.randn(NUM_ENVS, HISTORY_LENGTH * NUM_OBS)

        for step_i in range(num_trans):
            actions = alg.act(obs, priv, history)
            rewards = torch.randn(NUM_ENVS) * 0.1
            dones = torch.zeros(NUM_ENVS, dtype=torch.long)
            if step_i == 3:
                dones[0] = 1
                dones[5] = 1
            if step_i == 6:
                dones[:] = 1
            infos = {}
            alg.process_env_step(rewards, dones, infos)

        alg.compute_returns(priv, history)
        losses = alg.update()
        for loss in losses:
            assert not (loss != loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
