# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
import math
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from unitree_rl_lab.rl.actor_critic_moe_cts import ActorCriticMoECTS
from unitree_rl_lab.rl.moe_cts import MoECTS


class OnPolicyRunnerCTS:

    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.device = device
        self.env = env

        # Isaac Lab 2.2.0 interface
        self.num_envs = env.unwrapped.num_envs
        obs_dict = env.unwrapped.observation_manager.compute()
        num_obs = obs_dict["policy"].shape[-1]
        if "critic" not in obs_dict or obs_dict["critic"] is None:
            raise ValueError(
                "MoE+CTS requires critic observations (privileged obs) to be defined. "
                "Make sure the environment config includes a 'critic' observation group "
                "with height_scan and other privileged observations."
            )
        num_privileged_obs = obs_dict["critic"].shape[-1]
        num_actions = sum(env.unwrapped.action_manager.action_term_dim)

        # Support both flat dict (from MoECTSRunnerCfg.to_dict()) and nested dict formats
        if "runner" in train_cfg:
            self.cfg = train_cfg["runner"]
            self.alg_cfg = train_cfg["algorithm"]
            self.policy_cfg = train_cfg["policy"]
            history_length = train_cfg["history_length"]
        else:
            # Flat dict format from MoECTSRunnerCfg.to_dict()
            self.cfg = train_cfg
            self.alg_cfg = train_cfg["algorithm"]
            self.policy_cfg = train_cfg["policy"]
            history_length = train_cfg["history_length"]

        actor_critic_class = ActorCriticMoECTS
        model = actor_critic_class(
            num_obs,
            num_privileged_obs,
            num_actions,
            self.num_envs,
            history_length,
            **self.policy_cfg).to(self.device)
        alg_class = MoECTS
        self.alg = alg_class(model, self.num_envs, history_length, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.num_envs, self.num_steps_per_env, [num_obs], [num_privileged_obs], [num_actions])

        # init history
        self.history = torch.zeros((self.num_envs, history_length, num_obs), device=self.device)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        obs_dict, _ = env.unwrapped.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.unwrapped.episode_length_buf = torch.randint_like(
                self.env.unwrapped.episode_length_buf,
                high=int(self.env.unwrapped.max_episode_length))
        obs_dict, _ = self.env.unwrapped.reset()
        obs = obs_dict["policy"].to(self.device)
        privileged_obs = obs_dict["critic"].to(self.device)
        assert privileged_obs is not None, "Critic observations (privileged obs) must be defined for MoE+CTS training"
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        self.alg.model.train() # switch to train mode (for dropout for example)

        ep_infos = []
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.start_learning_iteration = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, self.history.flatten(1))
                    # Isaac Lab 2.2.0: clip actions (same as RslRlVecEnvWrapper)
                    actions = torch.clamp(actions, -100.0, 100.0)
                    # Isaac Lab 2.2.0: env.unwrapped.step returns 5-tuple
                    obs_dict, rewards, terminated, truncated, extras = self.env.unwrapped.step(actions)
                    obs = obs_dict["policy"].to(self.device)
                    privileged_obs = obs_dict["critic"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = (terminated | truncated).long().to(self.device)
                    extras = extras or {}
                    # Isaac Lab 2.2.0: time_outs for bootstrapping
                    # Only set if not already provided by the environment
                    if 'time_outs' not in extras:
                        extras['time_outs'] = truncated.to(self.device)
                    self.history[dones > 0] = 0.0
                    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
                    self.alg.process_env_step(rewards, dones, extras)

                    if self.log_dir is not None:
                        # Isaac Lab 2.2.0: episode info is in extras["log"], not extras["episode"]
                        if 'log' in extras:
                            ep_infos.append(extras['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.shape[0]:
                            ti = self.alg.teacher_env_idxs
                            teacher_ids = new_ids[torch.isin(new_ids, ti)]
                            student_ids = new_ids[~torch.isin(new_ids, ti)]
                            teacher_rewbuffer.extend(cur_reward_sum[teacher_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(privileged_obs, self.history.flatten(1))

            try:
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss = self.alg.update()
            except RuntimeError as e:
                # Catch CUDA OOM or NaN errors - save checkpoint and re-raise
                print(f"[ERROR] Update step failed at iteration {it}: {e}")
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, 'model_emergency_{}.pt'.format(it)), it, False)
                    print(f"[INFO] Emergency checkpoint saved to model_emergency_{it}.pt")
                raise

            # Check for NaN in losses (early detection of training divergence)
            if any(math.isnan(x) or math.isinf(x) for x in [mean_value_loss, mean_surrogate_loss]):
                print(f"[WARNING] NaN detected in losses at iteration {it}: "
                      f"value_loss={mean_value_loss}, surrogate_loss={mean_surrogate_loss}")
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, 'model_nan_{}.pt'.format(it)), it, False)
                    print(f"[INFO] Pre-NaN checkpoint saved to model_nan_{it}.pt")

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration += 1
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it, False)
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), it, True)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # Isaac Lab 2.2.0: extras["log"] values are scalars, not tensors
                    # Use .get() because different episodes may have different keys
                    val = ep_info.get(key)
                    if val is None:
                        continue
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor([val], device=self.device)
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(self.device)))
                if infotensor.shape[0] > 0:
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.model.std.mean()
        fps = int(self.num_steps_per_env * self.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/latent', locs['mean_latent_loss'], locs['it'])
        self.writer.add_scalar('Loss/load_balance', locs['mean_load_balance_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['teacher_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_teacher_reward', statistics.mean(locs['teacher_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_episode_length', statistics.mean(locs['teacher_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_reward/time', statistics.mean(locs['teacher_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_teacher_episode_length/time', statistics.mean(locs['teacher_lenbuffer']), self.tot_time)
        if len(locs['student_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_student_reward', statistics.mean(locs['student_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_episode_length', statistics.mean(locs['student_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_reward/time', statistics.mean(locs['student_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_student_episode_length/time', statistics.mean(locs['student_lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                      'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                      f"""{'Latent loss:':>{pad}} {locs['mean_latent_loss']:.4f}\n"""
                      f"""{'Load balance loss:':>{pad}} {locs['mean_load_balance_loss']:.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        if len(locs['teacher_rewbuffer']):
            log_string += (f"""{'Mean teacher reward:':>{pad}} {statistics.mean(locs['teacher_rewbuffer']):.2f}\n"""
                           f"""{'Mean teacher episode length:':>{pad}} {statistics.mean(locs['teacher_lenbuffer']):.2f}\n""")
        if len(locs['student_rewbuffer']):
            log_string += (f"""{'Mean student reward:':>{pad}} {statistics.mean(locs['student_rewbuffer']):.2f}\n"""
                           f"""{'Mean student episode length:':>{pad}} {statistics.mean(locs['student_lenbuffer']):.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration - self.start_learning_iteration) * (
                               locs['tot_iter'] - locs['it']):.1f}s\n""")
        print(log_string, flush=True)

    def save(self, path, it, last_model, infos=None):
        torch.save({
            'model_state_dict': self.alg.model.state_dict(),
            'optimizer1_state_dict': self.alg.optimizer1.state_dict(),
            'optimizer2_state_dict': self.alg.optimizer2.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.model.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer1.load_state_dict(loaded_dict['optimizer1_state_dict'])
            self.alg.optimizer2.load_state_dict(loaded_dict['optimizer2_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.model.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.model.to(device)
        return self.alg.model.act_inference
