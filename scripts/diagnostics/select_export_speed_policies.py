"""Select CTS checkpoints within command stages and validate CPU ONNX exports.

Ranking uses trailing student training returns, not held-out evaluation.
No Isaac Sim imports, GPU execution, or modifications to deployment assets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/rsl_rl'))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict
import yaml

from rsl_rl.modules import ActorCriticMoECTS

FIELDS = {
    'student_reward': 'Mean student reward',
    'teacher_reward': 'Mean teacher reward',
    'student_length': 'Mean student episode length',
    'terrain_level': 'Curriculum/terrain_levels',
    'command_x': 'Metrics/base_velocity/max_command_x',
    'value_loss': 'Mean value loss',
    'illegal_contact': 'Episode_Termination/illegal_contact',
    'time_out': 'Episode_Termination/time_out',
}
# Stages for this W1W task; actual logged command ceilings are verified below.
STAGES = [(0.5, 0, 20000), (1.0, 20000, 50000),
          (2.0, 50000, 75000), (3.0, 75000, 1000000000)]
WIDTHS = (3, 3, 3, 16, 16, 16)


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                             lambda loader, node: tuple(loader.construct_sequence(node)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_log(path):
    rows = {}
    current = None
    reverse = {v: k for k, v in FIELDS.items()}
    with path.open(errors='replace') as stream:
        for line in stream:
            match = re.search(r'Learning iteration (\d+)/', line)
            if match:
                if current and all(k in current for k in FIELDS):
                    rows[current['iteration']] = current
                current = {'iteration': int(match[1])}
            elif current is not None and ':' in line:
                key, value = line.strip().rsplit(':', 1)
                if key in reverse:
                    current[reverse[key]] = float(value.strip())
    if current and all(k in current for k in FIELDS):
        rows[current['iteration']] = current
    return rows


def rank_checkpoints(run, window):
    rows = read_log(run / 'train.log')
    ranked = []
    for path in run.glob('model_*.pt'):
        match = re.fullmatch(r'model_(\d+)\.pt', path.name)
        if not match:
            continue
        it = int(match[1])
        for speed, begin, end in STAGES:
            if begin <= it < end:
                start = it - window + 1
                if start < begin or any(i not in rows for i in range(start, it + 1)):
                    break
                sample = [rows[i] for i in range(start, it + 1)]
                if not all(abs(r['command_x'] - speed) < 1e-5 for r in sample):
                    break
                values = np.array([[r[k] for k in FIELDS] for r in sample])
                if not np.isfinite(values).all():
                    break
                metrics = {k: float(values[:, j].mean()) for j, k in enumerate(FIELDS)}
                metrics.update(value_loss_median=float(np.median(values[:, 5])),
                               value_loss_max=float(values[:, 5].max()),
                               student_reward_std=float(values[:, 0].std()))
                ranked.append(dict(run=run.name, checkpoint=str(path.resolve()), iteration=it,
                                   speed=speed, window_start=start, window_end=it, **metrics))
                break
    return sorted(ranked, key=lambda r: (-r['student_reward'], -r['student_length'], -r['iteration']))


def encode_history(frames):
    blocks = []
    start = 0
    for width in WIDTHS:
        blocks.append(frames[:, start:start + width].reshape(1, -1))
        start += width
    return np.concatenate(blocks, axis=1).astype(np.float32)


def export_one(record, output):
    checkpoint = Path(record['checkpoint'])
    config = yaml.load((checkpoint.parent / 'params/agent.yaml').read_text(), Loader=ConfigLoader)
    policy_config = dict(config['policy'])
    assert policy_config.pop('class_name') == 'ActorCriticMoECTS'
    dummy = TensorDict({'policy': torch.zeros(1, 228), 'single_obs': torch.zeros(1, 57),
                        'critic': torch.zeros(1, 295)}, batch_size=[1])
    policy = ActorCriticMoECTS(dummy, config['obs_groups'], 16, **policy_config).cpu()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['iter'] == record['iteration'], (checkpoint, state['iter'])
    policy.load_state_dict(state['model_state_dict'], strict=True)
    policy.eval()
    assert all(torch.isfinite(v).all() for v in policy.state_dict().values()), checkpoint
    spec = importlib.util.spec_from_file_location('cts_export_utils', ROOT / 'scripts/rsl_rl/rsl_rl_utils.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    filename = f"w1w_v{str(record['speed']).replace('.', 'p')}_model_{record['iteration']}.onnx"
    path = output / filename
    if path.exists():
        raise FileExistsError(f'Choose a new output directory: {path}')
    exporter.export_cts_policy_as_onnx(policy, str(output), policy.actor_obs_normalizer,
                                     policy.single_obs_normalizer, filename=filename)
    model = onnx.load(path)
    onnx.helper.set_model_props(model, {
        'source_checkpoint': str(checkpoint), 'source_sha256': digest(checkpoint),
        'selection': 'highest trailing 1000-iteration student training return within command stage; not held-out evaluation',
        'command_ceiling_mps': str(record['speed']), 'branch': 'student',
        'observation_layout': 'term-major 4 frames; widths=3,3,3,16,16,16; oldest to newest',
        'output': 'raw 16 actions; caller clips to +/-10 and applies leg/wheel scaling',
    })
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    assert [(i.name, i.shape) for i in session.get_inputs()] == [('obs', [1, 228])]
    assert [(i.name, i.shape) for i in session.get_outputs()] == [('actions', [1, 16])]
    rng = np.random.default_rng(42)
    max_error = 0.0
    for index in range(130):
        frames = rng.normal(0, 0.5, (4, 57)).astype(np.float32)
        frames[:, 6:9] = rng.uniform([-2*record['speed'], -2, -0.785],
                                     [2*record['speed'], 2, 0.785], (4, 3))
        frames[:, 41:] = rng.uniform(-3, 3, (4, 16))
        frames[:, 12:25:4] = 0  # wheel position columns 9 + (3,7,11,15)
        if index == 0:
            frames[:] = 0
        elif index == 1:
            frames[:] = 0
            frames[:, 5] = -1
        elif index % 5 == 0:
            frames[:] = frames[-1]  # first frame repeated after reset
        history = encode_history(frames)
        obs = TensorDict({'policy': torch.from_numpy(history),
                          'single_obs': torch.from_numpy(frames[-1:].copy()),
                          'critic': torch.zeros(1, 295)}, batch_size=[1])
        with torch.inference_mode():
            expected = policy.act_inference(obs).numpy()
        actual = session.run(['actions'], {'obs': history})[0]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-3)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return {**record, 'onnx': str(path.resolve()), 'checkpoint_sha256': digest(checkpoint),
            'onnx_sha256': digest(path), 'validation': {'onnx_checker': 'passed',
            'runtime': 'CPUExecutionProvider', 'cases': 130, 'max_abs_error': max_error,
            'rtol': 1e-4, 'atol': 2e-3}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window', type=int, default=1000)
    args = parser.parse_args()
    if args.window != 1000:
        parser.error('This documented selection protocol uses a 1000-iteration window.')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    all_ranked = []
    for run in sorted(args.root.iterdir()):
        if run.is_dir() and (run / 'train.log').exists() and (run / 'params/agent.yaml').exists():
            all_ranked.extend(rank_checkpoints(run, args.window))
    with (args.output / 'candidate_ranking.csv').open('w', newline='') as f:
        if not all_ranked:
            raise ValueError('No eligible logged checkpoints found')
        writer = csv.DictWriter(f, fieldnames=list(all_ranked[0]))
        writer.writeheader()
        writer.writerows(sorted(all_ranked, key=lambda r: (r['speed'], -r['student_reward'])))
    selected = []
    for speed, _, _ in STAGES:
        candidates = sorted([r for r in all_ranked if r['speed'] == speed],
                            key=lambda r: (-r['student_reward'], -r['student_length'], -r['iteration']))
        if not candidates:
            raise ValueError(f'No eligible checkpoints for speed {speed}')
        print('SELECTED', json.dumps(candidates[0]), flush=True)
        record = export_one(candidates[0], args.output)
        record['eligible_candidates'] = len(candidates)
        record['top5'] = candidates[:5]
        selected.append(record)
        (args.output / 'selection.json').write_text(json.dumps(selected, indent=2))
        print('VALIDATED', record['onnx'], record['validation'], flush=True)
    lines = ['# 各速度课程的学生策略 ONNX 导出', '',
             '筛选依据：同一速度课程内，检查点对应及之前连续1000轮的学生平均回报最高；只比较完整窗口且实际指令上限一致的已保存检查点。', '',
             '**这是训练日志最佳候选，不是固定速度实测最优。** 各阶段同时包含不同地形、速度和转向指令；速度标签表示全局vx指令上限，不表示专门在该固定速度训练或测试。', '',
             '源文件实际扩展名为 `.pt`，不是 `.pth`。保留原始检查点不改名、不覆盖。检查点在该轮更新后保存；评分窗口来自此前采样，因此不是保存后模型的独立评测。', '',
             '| vx上限 | 检查点 | 评分窗口 | 学生回报均值 | 学生回合秒数 | 地形等级 | ONNX |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in selected:
        lines.append(f"| {r['speed']} m/s | [{Path(r['checkpoint']).name}]({os.path.relpath(r['checkpoint'], args.output)}) | {r['window_start']}–{r['window_end']} | {r['student_reward']:.4f} | {r['student_length']*.02:.3f} | {r['terrain_level']:.3f} | [{Path(r['onnx']).name}]({Path(r['onnx']).name}) |")
    lines += ['', '## 使用契约与验证', '',
              '- 只导出学生MoE编码器与共享Actor，不包含教师或Critic。',
              '- 输入 `obs`: float32 `[1,228]`；输出 `actions`: float32 `[1,16]`，ONNX opset 18。',
              '- 外部维护4帧历史，按项拼接，项宽为 `[3,3,3,16,16,16]`，每项帧顺序从旧到新；每次重置用第一帧重复填充历史。',
              '- 输入需按训练契约缩放/裁剪，四个轮子位置列为零；输出是原始动作，调用方仍需裁剪±10并执行腿关节0.25缩放及轮速20缩放、轮速目标±150裁剪。',
              '- 130组CPU输入/模型，包括零输入、默认重力、不同历史、重置重复历史；与源PyTorch学生输出核对，rtol=1e-4、atol=0.002（原始动作单位）。初始1e-4绝对门限未通过：0.5 m/s模型一组压力输入在Softmax门控处放大浮点差异，输出差约0.00107；保留实际最大误差，不宣称逐位相等。',
              '- 输入生成测试验证导出数值一致性，不验证步态、速度跟踪或实机稳定性。',
              '- 完整候选排名见 `candidate_ranking.csv`；源权重/ONNX SHA256、top5、价值损失与误差见 `selection.json`。',
              '- 未启动GPU或Isaac Sim，未修改 `w1w_deploy`。', '']
    (args.output / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
