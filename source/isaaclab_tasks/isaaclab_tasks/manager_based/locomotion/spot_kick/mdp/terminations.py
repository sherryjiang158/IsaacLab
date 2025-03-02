# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def robot_fall(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate if robot falls (based on orientation)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Check if robot is too tilted
    projected_gravity = asset.data.projected_gravity_b
    too_tilted = torch.norm(projected_gravity[:, :2], dim=1) > 0.5
    return too_tilted

def successful_kick(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_velocity: float = 1.0,
    max_height: float = 0.5,
) -> torch.Tensor:
    """Terminate after successful kick (ball moving forward with good height)."""
    ball: RigidObject = env.scene[asset_cfg.name]
    
    # Check ball velocity and height
    forward_vel = ball.data.root_lin_vel_w[:, 0]
    height = ball.data.root_pos_w[:, 2]
    
    good_kick = (forward_vel > min_velocity) & (height < max_height)
    return good_kick

def ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    bounds: float = 3.0
) -> torch.Tensor:
    """Terminate if ball goes too far from robot."""
    ball: RigidObject = env.scene[asset_cfg.name]
    robot: RigidObject = env.scene["robot"]
    
    # Check if ball is too far from robot
    relative_pos = ball.data.root_pos_w - robot.data.root_pos_w
    too_far = torch.norm(relative_pos[:, :2], dim=1) > bounds
    return too_far

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate after max episode length."""
    return env.time_out