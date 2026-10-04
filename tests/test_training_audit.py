"""Regression tests for deployment observations and CTS training fixes."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
import weakref

import torch
from test_legbot_training_contract import load_functions, load_module, ROOT


class ObservationAndFailureTests(unittest.TestCase):
    def test_current_frame_and_wheel_placeholders(self):
        ns = load_functions('source/robot_lab/robot_lab/tasks/legbot/env/observations.py',
                            ['align_policy_observations'], {'torch': torch})
        obs = {'policy': torch.arange(456.).reshape(2, 228), 'single_obs': torch.ones(2, 57)}
        ns['align_policy_observations'](obs)
        start = 0
        expected = []
        for width in (3, 3, 3, 16, 16, 16):
            expected.append(obs['policy'][:, start:start+4*width].reshape(2, 4, width)[:, -1])
            start += 4*width
        torch.testing.assert_close(obs['single_obs'], torch.cat(expected, -1))
        self.assertEqual(obs['policy'][:, 36:100].reshape(2, 4, 16)[:, :, 3::4].count_nonzero(), 0)
        self.assertEqual(obs['single_obs'][:, 9:25][:, 3::4].count_nonzero(), 0)

    def test_smoothness_zero_is_a_valid_previous_action(self):
        ns = load_functions('source/robot_lab/robot_lab/tasks/go2/mdp/rewards.py',
                            ['action_smoothness_l2'], {'torch': torch})
        env = NS(episode_length_buf=torch.tensor([1, 2, 3]), action_manager=NS(
            action=torch.ones(3, 16), prev_action=torch.zeros(3, 16), prev_prev_action=torch.zeros(3, 16)))
        torch.testing.assert_close(ns['action_smoothness_l2'](env), torch.tensor([0., 0., 16.]))

    def command_fixture(self, *, zero_probability, limit_probability):
        util = load_functions('source/robot_lab/robot_lab/tasks/go2/mdp/utils.py',
                              ['sample_single_interval', 'sample_disjoint_intervals'], {'torch': torch})
        method = load_functions('source/robot_lab/robot_lab/tasks/go2/mdp/commands.py',
                                ['_resample'], util, 'Go2RLGymCommand')['_resample']
        n = 4
        env = NS(common_step_counter=0, max_episode_length=1250, max_episode_length_s=25.,
                 episode_length_buf=torch.zeros(n), step_dt=.02)
        term = NS(_env=env, device='cpu', cfg=NS(
            command_range_curriculum=[], num_steps_per_iter=24, resampling_time=5.,
            dynamic_resample_commands=False, limit_vel_prob=limit_probability,
            limit_vel_invert_when_continuous=True, zero_command_curriculum={},
            limit_ang_vel_at_zero_command_prob=0.),
            commands_xy_accumulation=torch.zeros(n,2), terrain_length=8., time_left=torch.zeros(n),
            commands=torch.tensor([[.1,.2,.3]]*n), env_command_ranges={
                key: torch.tensor([[-1.,1.]]*n) for key in ('lin_vel_x','lin_vel_y','ang_vel_yaw')},
            last_is_limit_vel=torch.ones(n,dtype=torch.bool), limit_vel_comb=torch.tensor([[-1,-1,0],[1,1,1]]),
            max_lin_vel=1., zero_command_prob=zero_probability,
            get_current_scale=lambda cfg: zero_probability)
        return method, term

    def test_zero_commands_do_not_keep_random_yaw(self):
        method, term = self.command_fixture(zero_probability=1., limit_probability=0.)
        method(term, torch.arange(4))
        torch.testing.assert_close(term.commands, torch.zeros(4,3))

    def test_continuous_limit_reverses_previous_command(self):
        method, term = self.command_fixture(zero_probability=0., limit_probability=1.)
        previous = term.commands.clone()
        method(term, torch.arange(4))
        torch.testing.assert_close(term.commands, -previous)

    def test_fatal_gpu_failure_exits_with_saved_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            code = (
                f"import sys; sys.path.insert(0, {str(ROOT / 'scripts/rsl_rl')!r}); "
                "from simulator_failures import exit_on_fatal_simulator_error; "
                f"exit_on_fatal_simulator_error(RuntimeError('Failed to get DOF velocities from backend'), {directory!r}, 'cuda:0'); "
                "raise SystemExit(99)"
            )
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Failed to get DOF', (Path(directory) / 'fatal_error.log').read_text())
        module = load_module('sim_failures', 'scripts/rsl_rl/simulator_failures.py')
        module.exit_on_fatal_simulator_error(ValueError('bad configuration'), '/does/not/exist', 'cuda:0')


@unittest.skipUnless(importlib.util.find_spec('tensordict') and importlib.util.find_spec('rsl_rl'),
                     'Requires the training environment (rsl_rl and tensordict)')
class CTSAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_algorithm(self, n=8, ratio=.75, epochs=3):
        from tensordict import TensorDict
        from rsl_rl.modules import ActorCriticMoECTS
        from rsl_rl.algorithms import MoECTS
        from rsl_rl.storage import RolloutStorageCTS
        obs = TensorDict({'policy': torch.randn(n, 12), 'single_obs': torch.randn(n, 3),
                          'critic': torch.randn(n, 8)}, batch_size=[n])
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCriticMoECTS(obs, {'policy': ['policy'], 'critic': ['critic']}, 2,
                                      expert_num=2, latent_dim=4, teacher_encoder_hidden_dims=[8],
                                      student_encoder_hidden_dims=[8], actor_hidden_dims=[8], critic_hidden_dims=[8])
        count = min(max(int(n*ratio), 1), n-1)
        storage = RolloutStorageCTS('rl', n, count, 4, obs, [2])
        algo = MoECTS(policy, storage, n, teacher_env_ratio=ratio, num_learning_epochs=epochs,
                     num_mini_batches=2, schedule='fixed')
        return algo, obs

    def test_arbitrary_teacher_splits(self):
        for n, ratio in [(8, .75), (5, .75), (9, .6), (2, .01), (2048, .75)]:
            with self.subTest(n=n, ratio=ratio):
                algo, _ = self.make_algorithm(n, ratio)
                all_ids = torch.cat((algo.teacher_env_idxs, algo.student_env_idxs)).sort().values
                torch.testing.assert_close(all_ids, torch.arange(n))
                self.assertEqual(len(algo.teacher_env_idxs), min(max(int(n*ratio), 1), n-1))
        algo, _ = self.make_algorithm(8, .75)
        torch.testing.assert_close(algo.student_env_idxs, torch.tensor([0, 4]))

    def test_streaming_update_and_student_learning(self):
        algo, obs = self.make_algorithm()
        before = [p.detach().clone() for p in algo.policy.student_moe_encoder.parameters()]
        with torch.inference_mode():
            for _ in range(4):
                actions = algo.act(obs)
                self.assertEqual(tuple(actions.shape), (8, 2))
                algo.process_env_step(obs, torch.randn(8), torch.zeros(8), {})
            algo.compute_returns(obs)
        original = algo.storage.mini_batch_generator
        live = []
        peak = [0]
        def monitored(*args):
            for batch in original(*args):
                live[:] = [ref for ref in live if ref() is not None]
                live.append(weakref.ref(batch[0]))
                peak[0] = max(peak[0], len(live))
                yield batch
        algo.storage.mini_batch_generator = monitored
        losses = algo.update()
        self.assertLessEqual(peak[0], 2, 'All epochs must not be retained on GPU')
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        self.assertEqual(algo.storage.step, 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, algo.policy.student_moe_encoder.parameters())))

    def test_negative_scalar_std_stays_finite(self):
        algo, obs = self.make_algorithm()
        with torch.no_grad():
            algo.policy.std.fill_(-.1)
            action = algo.policy.act(obs, is_teacher=False)
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue((algo.policy.action_std > 0).all())
