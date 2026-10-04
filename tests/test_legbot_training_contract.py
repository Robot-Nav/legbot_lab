"""CPU regressions. Isaac-dependent functions are AST-loaded without starting Sim."""
import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_functions(relative, names, namespace, class_name=None):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(ROOT / relative), "exec"), namespace)
    return namespace


class Encoder(torch.nn.Module):
    def forward(self, obs: torch.Tensor):
        return obs[:, :32], obs[:, :1]


class TrainingContractTests(unittest.TestCase):
    def test_joint_selection(self):
        ns = load_functions("source/robot_lab/robot_lab/tasks/go2/mdp/observations.py",
                            ["joint_pos_rel_without_wheel"], {"torch": torch, "SceneEntityCfg": lambda name: NS(name=name)})
        values = torch.arange(1, 17).reshape(1, 16).float()
        env = NS(scene={"robot": NS(data=NS(joint_pos=values, default_joint_pos=torch.zeros_like(values)))})
        cases = [(slice(None), [3, 7, 11, 15]), (slice(1, 16, 2), slice(3, 16, 4)),
                 ([7, 2, 3, 0], [3, 7, 11, 15]), (slice(None), slice(None))]
        for selected, wheels in cases:
            with self.subTest(selected=selected, wheels=wheels):
                result = ns["joint_pos_rel_without_wheel"](env, NS(name="robot", joint_ids=selected), NS(joint_ids=wheels))
                ids = list(range(16)[selected]) if isinstance(selected, slice) else selected
                wheel_ids = list(range(16)[wheels]) if isinstance(wheels, slice) else wheels
                expected = torch.tensor([[0 if i in wheel_ids else i + 1 for i in ids]], dtype=torch.float32)
                torch.testing.assert_close(result, expected)
        torch.testing.assert_close(values, torch.arange(1, 17).reshape(1, 16).float())

    def test_both_jit_exporters(self):
        files = ["source/rsl_rl/rsl_rl/utils/exporter_cts.py", "scripts/rsl_rl/rsl_rl_utils.py"]
        for index, file in enumerate(files):
            with self.subTest(file=file), tempfile.TemporaryDirectory() as directory:
                module = load_module(f"exporter_{index}", file)
                policy = NS(is_recurrent=False, actor=torch.nn.Linear(89, 16), student_moe_encoder=Encoder(),
                            state_dependent_std=False, num_actions=16, num_single_obs=57, num_actor_obs=228)
                exporter = module._TorchPolicyExporter(policy)
                exporter.export(directory, "policy.pt")
                jit = torch.jit.load(str(Path(directory) / "policy.pt"))
                first = torch.arange(57).float().reshape(1, 57)
                second = first + 100
                frames = [first, first, first, first]
                def check(frame):
                    self.assertEqual(tuple(jit(frame).shape), (1, 16))
                    expected = torch.cat([torch.cat([x[:, lo:hi] for x in frames], -1)
                                          for lo, hi in [(0,3),(3,6),(6,9),(9,25),(25,41),(41,57)]], -1)
                    torch.testing.assert_close(jit.obs_history, expected)
                check(first)
                frames = [first, first, first, second]
                check(second)
                jit.reset()
                frames = [second] * 4
                check(second)

    def runner_fixture(self):
        ns = load_functions("source/rsl_rl/rsl_rl/runners/on_policy_runner_cts.py", ["save", "load"],
                            {"torch": torch}, "OnPolicyRunnerCTS")
        class Runner:
            save = ns["save"]
            load = ns["load"]
        runner = Runner()
        policy = Mock()
        policy.load_state_dict.return_value = True
        policy.state_dict.return_value = {}
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
        runner.alg = NS(policy=policy, optimizer=optimizer, optimizer_stu_enc=optimizer)
        runner.alg_cfg = {"rnd_cfg": None}
        runner.cfg = {"num_steps_per_env": 24}
        runner.current_learning_iteration = 499
        runner.env = NS(unwrapped=NS(common_step_counter=12007))
        runner.env.reset = Mock(side_effect=lambda: self.assertEqual(runner.env.unwrapped.common_step_counter, self.expected_step))
        runner.logger = Mock()
        runner.update_robogauge = Mock()
        return runner

    def test_checkpoint_round_trip(self):
        runner = self.runner_fixture()
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory) / "model.pt")
            runner.save(file, 499, False)
            runner.alg.optimizer.param_groups[0]["lr"] = 1e-3
            runner.env.unwrapped.common_step_counter = 0
            self.expected_step = 12007
            runner.load(file)
        self.assertEqual(runner.current_learning_iteration, 500)
        self.assertEqual(runner.alg.learning_rate, 2e-4)
        runner.env.reset.assert_called_once()

    def test_legacy_checkpoint_progress(self):
        runner = self.runner_fixture()
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory) / "model.pt")
            torch.save({"model_state_dict": {}, "iter": 499, "infos": None}, file)
            self.expected_step = 12000
            runner.load(file, load_optimizer=False)
        self.assertEqual(runner.current_learning_iteration, 500)
        runner.env.reset.assert_called_once()

    def test_mujoco_targets(self):
        ns = load_functions("deploy/deploy_mujoco/deploy_legbot.py", ["action_to_targets"], {"np": np})
        mapping = np.array([15, 0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7])
        cfg = NS(default_angles=np.arange(16, dtype=float), idx_mj2model=mapping,
                 idx_model2mj=np.argsort(mapping), num_actions=16, action_pos_scale=.25, action_vel_scale=20.)
        action = np.array([100, -100, 2, 100] * 4, dtype=float)
        pos, vel = ns["action_to_targets"](action, cfg)
        pos_model, vel_model = pos[mapping], vel[mapping]
        for j in range(16):
            if j % 4 == 3:
                self.assertEqual(vel_model[j], 150.)
                self.assertEqual(pos_model[j], cfg.default_angles[mapping[j]])
            else:
                self.assertEqual(vel_model[j], 0.)
                self.assertEqual(pos_model[j], cfg.default_angles[mapping[j]] + np.clip(action[j], -10, 10) * .25)
        _, negative = ns["action_to_targets"](-action, cfg)
        np.testing.assert_array_equal(negative[mapping][3::4], [-150.] * 4)


if __name__ == "__main__":
    unittest.main()
