# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that can be used to enable Spot randomizations.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_min_pos = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    # clip vel to range
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    # sample these values randomly
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def randomize_ball_position(env, env_ids: torch.Tensor, position_range: tuple) -> None:
    """
    Randomizes the ball's starting position for variability.
    
    Parameters:
      env: The simulation environment.
      env_ids: A tensor of indices for the environments to randomize.
      position_range: A tuple of three tuples, one for each axis:
                      ((min_x, max_x), (min_y, max_y), (min_z, max_z))
                      These values are offsets added to the ball's default position.
    
    Explanation:
      - Retrieves the ball asset from the scene using its name ("ball").
      - Extracts the default position for the ball from asset.data.default_pos.
      - Computes per-axis minimum and maximum positions by adding the corresponding offset range.
      - Samples uniformly between these limits for each environment.
      - Resets the ball's velocity to zero and writes the new state back into the simulation.
    """
    # Get the ball asset from the environment
    asset = env.scene["ball"]
    
    # Assume the ball's default position is stored in asset.data.default_pos (shape [N, 3])
    default_pos = asset.data.default_pos[env_ids]
    
    # Compute per-axis minimum and maximum positions based on the given offsets.
    min_x = default_pos[:, 0] + position_range[0][0]
    max_x = default_pos[:, 0] + position_range[0][1]
    min_y = default_pos[:, 1] + position_range[1][0]
    max_y = default_pos[:, 1] + position_range[1][1]
    min_z = default_pos[:, 2] + position_range[2][0]
    max_z = default_pos[:, 2] + position_range[2][1]
    
    # Stack to form tensors of shape [N, 3] for minimum and maximum positions.
    min_pos = torch.stack([min_x, min_y, min_z], dim=-1)
    max_pos = torch.stack([max_x, max_y, max_z], dim=-1)
    
    # Sample uniformly between these limits. We use the provided sample_uniform helper.
    new_pos = sample_uniform(min_pos, max_pos, default_pos.shape, default_pos.device)
    
    # Set the new velocity to zero.
    new_vel = torch.zeros_like(new_pos)
    
    # Write the new state into the simulation for the ball asset.
    asset.write_state_to_sim(new_pos, new_vel, env_ids=env_ids)
