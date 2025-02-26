# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task Rewards
##


# class GaitReward(ManagerTermBase):
#     """Gait enforcing reward term for quadrupeds.

#     This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
#     to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
#     quadrupedal gaits with two pairs of synchronized feet.
#     """

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         """Initialize the term.

#         Args:
#             cfg: The configuration of the reward.
#             env: The RL environment instance.
#         """
#         super().__init__(cfg, env)
#         self.std: float = cfg.params["std"]
#         self.max_err: float = cfg.params["max_err"]
#         self.velocity_threshold: float = cfg.params["velocity_threshold"]
#         self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#         # match foot body names with corresponding foot body ids
#         synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
#         if (
#             len(synced_feet_pair_names) != 2
#             or len(synced_feet_pair_names[0]) != 2
#             or len(synced_feet_pair_names[1]) != 2
#         ):
#             raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
#         synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
#         synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
#         self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         std: float,
#         max_err: float,
#         velocity_threshold: float,
#         synced_feet_pair_names,
#         asset_cfg: SceneEntityCfg,
#         sensor_cfg: SceneEntityCfg,
#     ) -> torch.Tensor:
#         """Compute the reward.

#         This reward is defined as a multiplication between six terms where two of them enforce pair feet
#         being in sync and the other four rewards if all the other remaining pairs are out of sync

#         Args:
#             env: The RL environment instance.
#         Returns:
#             The reward value.
#         """
#         # for synchronous feet, the contact (air) times of two feet should match
#         sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
#         sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
#         sync_reward = sync_reward_0 * sync_reward_1
#         # for asynchronous feet, the contact time of one foot should match the air time of the other one
#         async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
#         async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
#         async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
#         async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
#         async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
#         # only enforce gait if cmd > 0
#         cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
#         body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
#         return torch.where(
#             torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
#         )

#     """
#     Helper functions.
#     """

#     def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
#         """Reward synchronization of two feet."""
#         air_time = self.contact_sensor.data.current_air_time
#         contact_time = self.contact_sensor.data.current_contact_time
#         # penalize the difference between the most recent air time and contact time of synced feet pairs.
#         se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
#         se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
#         return torch.exp(-(se_air + se_contact) / self.std)

#     def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
#         """Reward anti-synchronization of two feet."""
#         air_time = self.contact_sensor.data.current_air_time
#         contact_time = self.contact_sensor.data.current_contact_time
#         # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
#         # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
#         se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
#         se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
#         return torch.exp(-(se_act_0 + se_act_1) / self.std)

# ########################################################

class KickingGaitReward(ManagerTermBase):
    """Reward term for maintaining proper kicking stance.
    
    Ensures:
    1. Kicking foot (front right) is free to move
    2. Other three feet maintain ground contact
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # Define foot body IDs (example - adjust based on your URDF)
        # These would be the indices in the URDF for each foot
        self.kicking_foot_id = self.contact_sensor.find_bodies(["front_right_foot"])[0]
        self.support_feet_ids = self.contact_sensor.find_bodies(
            ["front_left_foot", "rear_left_foot", "rear_right_foot"]
        )[0]
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward."""
        # Check support feet contact
        support_contacts = self.contact_sensor.data.net_forces_w_history[:, :, self.support_feet_ids]
        support_in_contact = torch.norm(support_contacts, dim=-1) > 1.0
        support_reward = torch.all(support_in_contact, dim=-1).float()
        
        # Check kicking foot clearance when preparing kick
        foot_height = self.asset.data.body_pos_w[:, self.kicking_foot_id, 2]
        clearance_reward = torch.exp(-(foot_height - 0.1)**2 / 0.01)  # Reward ~10cm clearance
        
        return support_reward * clearance_reward


def ball_forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward forward velocity of the ball after kick."""
    ball = env.scene[asset_cfg.name]
    ball_vel = ball.data.root_lin_vel_w
    forward_vel = ball_vel[:, 0]  # Assuming x is forward
    return forward_vel

def ball_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    min_height: float = 0,
    max_height: float = 0.5,
) -> torch.Tensor:
    """Reward keeping ball within desired height range."""
    ball = env.scene[asset_cfg.name]
    ball_height = ball.data.root_pos_w[:, 2]
    in_range = (ball_height > min_height) & (ball_height < max_height)
    return in_range.float()

def kick_impact_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward quality of kick impact."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    ball = env.scene[ball_cfg.name]
    
    # Get impact force and direction
    impact_force = contact_sensor.data.net_forces_w_history[:, -1, foot_cfg.body_ids]
    impact_direction = ball.data.root_lin_vel_w / (torch.norm(ball.data.root_lin_vel_w, dim=-1, keepdim=True) + 1e-6)
    
    # Reward based on force magnitude and direction
    force_quality = torch.exp(-torch.square(torch.norm(impact_force, dim=-1) - 20.0) / 5.0)
    direction_quality = impact_direction[:, 0]  # Reward forward direction
    
    return force_quality * direction_quality

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def kick_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward good contact between foot and ball."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ball = env.scene[asset_cfg.name]
    
    # Get contact forces between kicking foot and ball
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact_magnitude = torch.norm(contact_forces, dim=-1)
    
    # Reward based on contact force (could be tuned)
    good_contact = (contact_magnitude > 5.0) & (contact_magnitude < 50.0)
    return good_contact.float()

def kick_foot_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_velocity: float = 2.0,
    std: float = 0.5
) -> torch.Tensor:
    """Reward kicking foot velocity."""
    # Get kicking foot ID (front right foot)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    kicking_foot_id = contact_sensor.find_bodies(["front_right_foot"])[0]
    
    # Get foot velocity
    asset = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, kicking_foot_id]
    vel_error = torch.square(torch.norm(foot_vel, dim=-1) - target_velocity)
    return torch.exp(-vel_error / std)

def support_feet_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize if support feet lose contact or have excessive force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get support feet IDs
    support_feet_ids = contact_sensor.find_bodies(
        ["front_left_foot", "rear_left_foot", "rear_right_foot"]
    )[0]
    
    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, support_feet_ids]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    
    # Check if all support feet are in contact
    in_contact = force_magnitudes > 1.0
    all_feet_contact = torch.all(in_contact, dim=-1)
    
    # Check if forces are within reasonable range
    excessive_force = force_magnitudes > force_threshold
    any_excessive = torch.any(excessive_force, dim=-1)
    
    return all_feet_contact.float() * (~any_excessive).float()


##
# Regularization Penalties
##

# def balance_penalty(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Penalize excessive roll and pitch of robot during kick."""
#     robot = env.scene[asset_cfg.name]
#     # Get roll and pitch from gravity projection
#     gravity = robot.data.projected_gravity_b
#     roll_pitch = torch.norm(gravity[:, :2], dim=1)
#     return roll_pitch

def support_feet_leave_ground_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    penalty_scale: float = 5.0,
) -> torch.Tensor:
    """Heavily penalize if any of the support feet (non-kicking feet) leave the ground.
    
    Args:
        env: Environment instance
        asset_cfg: Robot asset configuration
        sensor_cfg: Contact sensor configuration
        penalty_scale: How severe the penalty should be
        
    Returns:
        torch.Tensor: Penalty value (negative reward) if support feet leave ground
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get support feet IDs (all except front right foot which is kicking)
    support_feet_ids = contact_sensor.find_bodies(
        ["front_left_foot", "rear_left_foot", "rear_right_foot"]
    )[0]
    
    # Get contact forces for support feet
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, support_feet_ids]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    
    # Check if any support foot loses contact (force near zero)
    feet_in_contact = force_magnitudes > 1.0  # threshold of 1N
    all_support_feet_in_contact = torch.all(feet_in_contact, dim=-1)
    
    # Return penalty (negative reward) if any support foot leaves ground
    return -penalty_scale * (~all_support_feet_in_contact).float()

def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


# def joint_position_penalty(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
# ) -> torch.Tensor:
#     """Penalize joint position error from default on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#     reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


# def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint torques on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.linalg.norm((asset.data.applied_torque), dim=1)


# def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint velocities on the articulation."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     return torch.linalg.norm((asset.data.joint_vel), dim=1)

