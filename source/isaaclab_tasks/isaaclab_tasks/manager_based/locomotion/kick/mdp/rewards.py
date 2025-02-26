# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

class KickingGaitReward(ManagerTermBase):
    """Reward term for maintaining proper kicking stance."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # Define foot body IDs
        self.kicking_foot_id = self.contact_sensor.find_bodies(["front_right_foot"])[0]
        self.support_feet_ids = self.contact_sensor.find_bodies(
            ["front_left_foot", "rear_left_foot", "rear_right_foot"]
        )[0]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # Check support feet contact
        support_contacts = self.contact_sensor.data.net_forces_w_history[:, :, self.support_feet_ids]
        support_in_contact = torch.norm(support_contacts, dim=-1) > 1.0
        support_reward = torch.all(support_in_contact, dim=-1).float()
        
        # Check kicking foot clearance
        foot_height = self.asset.data.body_pos_w[:, self.kicking_foot_id, 2]
        clearance_reward = torch.exp(-(foot_height - 0.1)**2 / 0.01)  # ~10cm clearance
        
        return support_reward * clearance_reward

def ball_forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_velocity: float = 2.0,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward forward velocity of the ball after kick."""
    ball: RigidObject = env.scene[asset_cfg.name]
    ball_vel = ball.data.root_lin_vel_w
    forward_vel = ball_vel[:, 0]  # x is forward
    vel_error = torch.square(forward_vel - target_velocity)
    return torch.exp(-vel_error / std**2)

def kick_impact_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    optimal_force: float = 20.0,
    force_std: float = 5.0,
) -> torch.Tensor:
    """Reward quality of kick impact."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    ball = env.scene[ball_cfg.name]
    
    # Get impact force and direction
    impact_force = contact_sensor.data.net_forces_w_history[:, -1, foot_cfg.body_ids]
    impact_direction = ball.data.root_lin_vel_w / (torch.norm(ball.data.root_lin_vel_w, dim=-1, keepdim=True) + 1e-6)
    
    # Reward based on force magnitude and direction
    force_quality = torch.exp(-torch.square(torch.norm(impact_force, dim=-1) - optimal_force) / force_std)
    direction_quality = impact_direction[:, 0]  # Reward forward direction
    
    return force_quality * direction_quality

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
    
    # Check stability conditions
    in_contact = force_magnitudes > 1.0
    all_feet_contact = torch.all(in_contact, dim=-1)
    excessive_force = force_magnitudes > force_threshold
    any_excessive = torch.any(excessive_force, dim=-1)
    
    return all_feet_contact.float() * (~any_excessive).float()

# Regularization Penalties
def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in actions."""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)

def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )

def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)

def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)