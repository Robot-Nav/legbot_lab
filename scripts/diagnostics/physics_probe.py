"""Bounded physics-only comparison: official tasks, W1W, terrain and self contacts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--task',default='Isaac-Velocity-Rough-Unitree-Go2-v0')
parser.add_argument('--num_envs',type=int,default=2048)
parser.add_argument('--steps',type=int,default=12000)
parser.add_argument('--flat',action='store_true')
parser.add_argument('--no_self_collision',action='store_true')
parser.add_argument('--preload_warp',default=None)
parser.add_argument('--output',default='/tmp/w1w_physics_probe')
parser.add_argument('--checkpoint',default=None)
parser.add_argument('--synchronize',action='store_true')
# Parse the optional Warp override before launching Kit, leaving other args to AppLauncher.
partial,_=parser.parse_known_args()
if partial.preload_warp:
    sys.path.insert(0,partial.preload_warp)
    import warp
    print('Preloaded Warp',warp.__version__,warp.__file__,flush=True)
import torch
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)
app=launcher.app
import gymnasium as gym
import warp
import isaaclab_tasks
import robot_lab.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.terrains import MeshPlaneTerrainCfg
from tensordict import TensorDict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from simulator_failures import exit_on_fatal_simulator_error

print(json.dumps({'warp':warp.__version__,'warp_path':warp.__file__,'torch':torch.__version__,
                  'task':args.task,'envs':args.num_envs,'flat':args.flat,
                  'self_collision_disabled':args.no_self_collision}),flush=True)
Path(args.output).mkdir(parents=True,exist_ok=True)
Path(args.output,'loaded_libraries.txt').write_text(''.join(
    line for line in Path('/proc/self/maps').read_text().splitlines(keepends=True)
    if any(x in line.lower() for x in ['libcuda','libnvrtc','/warp.so','physxgpu'])))
cfg=load_cfg_from_registry(args.task,'env_cfg_entry_point')
cfg.scene.num_envs=args.num_envs
cfg.sim.device=args.device or 'cuda:0'
cfg.seed=42
cfg.log_dir=args.output
if args.flat:
    cfg.scene.terrain.terrain_generator.sub_terrains={'flat':MeshPlaneTerrainCfg(proportion=1.)}
if args.no_self_collision:
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions=False
    if hasattr(cfg.scene.robot.spawn,'self_collision'): cfg.scene.robot.spawn.self_collision=False
env=None
step=-1
try:
    env=gym.make(args.task,cfg=cfg)
    obs,_=env.reset()
    base=env.unwrapped
    actor=None
    if args.checkpoint:
        from rsl_rl.modules import ActorCriticMoECTS
        agent_cfg=load_cfg_from_registry(args.task,'rsl_rl_cfg_entry_point')
        policy_cfg=agent_cfg.policy.to_dict(); policy_cfg.pop('class_name')
        actor=ActorCriticMoECTS(TensorDict(obs,batch_size=[args.num_envs]),agent_cfg.obs_groups,
                               base.action_manager.total_action_dim,**policy_cfg).to(base.device)
        checkpoint=torch.load(args.checkpoint,map_location=base.device,weights_only=False)
        actor.load_state_dict(checkpoint['model_state_dict']);actor.eval()
    actions=torch.zeros(env.action_space.shape,device=base.device)
    start=time.monotonic()
    for step in range(args.steps):
        with torch.inference_mode():
            if actor is not None:
                actions=actor.act_inference(TensorDict(obs,batch_size=[args.num_envs])).clamp(-10,10)
            obs,reward,terminated,truncated,extras=env.step(actions)
            if args.synchronize: torch.cuda.synchronize()
        if step%250==0:
            assert torch.isfinite(reward).all(), 'Nonfinite simulation reward'
            print(json.dumps({'step':step,'elapsed_s':round(time.monotonic()-start,1),
                              'mean_reward':float(reward.mean()),
                              'done':int((terminated|truncated).sum())}),flush=True)
    print(json.dumps({'passed':True,'steps':args.steps,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    env.close()
except Exception as error:
    print(json.dumps({'failed_step':step,'error':str(error)}),flush=True)
    exit_on_fatal_simulator_error(error,args.output,cfg.sim.device)
    raise
app.close()
