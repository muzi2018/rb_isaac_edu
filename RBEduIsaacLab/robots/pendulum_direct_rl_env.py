# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from .pendulum import PENDULUM_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 1.0
    action_scale = 1.0  # [N]
    action_space = 1
    observation_space = 2
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    pole_dof_name = "continuous_joint"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    initial_pole_angle_range = [-0.0, 0.0]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    rew_scale_alive = 0.0
    rew_scale_terminated = 0.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class CartpoleDirectRLEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._pole_dof_idx, _ = self._robot.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale
        self.max_action = 25.0

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        # TODO: Setup target pos and vel 
        self.target_pos = math.pi
        self.target_vel = 0.0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["cartpole"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        # absoluate value of action must be lower than self.max_action 
        self.actions = torch.clamp(self.actions, -self.max_action, self.max_action)

    def _apply_action(self) -> None:
        self._robot.set_joint_effort_target(self.actions, joint_ids=self._pole_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        self.pos_diff = self.joint_pos[:, self._pole_dof_idx[0]] - self.target_pos
        self.vel_diff = self.joint_vel[:, self._pole_dof_idx[0]] - self.target_vel

        pose_term = torch.square(self.pos_diff).sum(dim=-1)
        # print(f"{self.pos_diff=}")

        # self.joint_vel[:, self._pole_dof_idx[0]].size() : torch.Size([num_env])
        # self.actions.size() : torch.Size([num_env, 1])
        # self.actions[:, self._pole_dof_idx[0]].size() : torch.Size([num_env])
        
        # temp_val_1 = torch.sum(torch.square(self.pos_diff).unsqueeze(dim=1), dim=-1)
        # temp_val_2 = torch.sum(torch.square(self.pos_diff), dim=-1)
        # print(f"{temp_val_1.size()=} {temp_val_2.size()=}")
        # temp_val_1.size()=torch.Size([16]) temp_val_2.size()=torch.Size([])

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.pos_diff,
            self.vel_diff,
            self.actions[:, self._pole_dof_idx[0]],
            self.reset_terminated,
        )

        # print(f"{total_reward=}")
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        
        # Rotate over 2 pi => out of bounds
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi * 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pos_diff: torch.Tensor,
    vel_diff: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()

    reward_type = "open_ai_gym"
    reward = torch.zeros_like(reset_terminated)
    
    # Legacy part
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_pole_vel

    if reward_type == "continuous":
        reward = torch.sum(-pos_diff, dim=-1)
    elif reward_type == "discrete":
        state_target_epsilon = 1e-2
        reward = torch.sum((pos_diff < state_target_epsilon).float(), dim=-1)
    elif reward_type == "soft_binary":
        reward = torch.sum(torch.exp(-pos_diff ** 2 / (2 * 0.25 ** 2)), dim=-1)
    elif reward_type == "soft_binary_with_repellor":
        # Attraction to target
        reward = torch.sum(torch.exp(-pos_diff ** 2 / (2 * 0.25 ** 2)), dim=-1)

        # Repulsion from hanging down (angle 0)
        pos_diff_repellor = torch.norm(pole_pos - 0.0, dim=-1)
        reward -= torch.sum(torch.exp(-pos_diff_repellor ** 2 / (2 * 0.25 ** 2)), dim=-1)
    elif reward_type == "open_ai_gym":
        if actions is None:
            raise ValueError("Action tensor is required for 'open_ai_gym' reward type.")
        # Pose penalty
        pose_scale = -1.0
        reward = pose_scale * torch.sum(torch.square(pos_diff).unsqueeze(dim=1), dim=-1)

        # Velocity penalty / pole_vel size : torch.Size([num_env])
        vel_scale = -0.1
        reward += vel_scale * torch.sum(torch.square(pole_vel).unsqueeze(dim=1), dim=-1)

        # Action/torque penalty / actions size : torch.Size([num_env])
        act_scale = -0.001
        reward += act_scale * torch.sum(torch.square(actions).unsqueeze(dim=1), dim=-1)
    elif reward_type == "open_ai_gym_red_torque":
        if actions is None:
            raise ValueError("Action tensor is required for 'open_ai_gym_red_torque' reward type.")
        pose_term = torch.square(pos_diff).sum(dim=-1)
        vel_term = torch.square(pole_vel).sum(dim=-1)
        act_term = torch.square(actions).sum(dim=-1)
        reward = -pose_term - 0.1 * vel_term - 0.01 * act_term
        # reward = -pos_diff ** 2 - 0.1 * vel_diff ** 2 - 0.01 * torch.sum(actions ** 2, dim=-1)
    else:
        # TorchScript doesn't like exceptions, so return NaNs instead
        reward = torch.full_like(reset_terminated, float('nan'))

    total_reward = reward + rew_alive + rew_termination
    return total_reward