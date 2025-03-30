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
    # print("reset JOINT!!!!!")
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    # default_joint_pos quantity is configured through the isaaclab.assets.ArticulationCfg.init_state parameter.
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

def randomize_ball_position(env, position_range: tuple = None) -> None:
    """
    Positions the ball next to the robot's kicking foot (front right) with optional random perturbations,
    and sets a default orientation and zero angular velocity.

    This function:
      1. Retrieves the toe position from the kicking leg frame.
      2. Adds a base offset to place the ball next to the toe.
      3. Optionally adds random offsets (if position_range is provided) for variability.
      4. Sets the ball's orientation (quaternion) to a default value (identity) and resets its velocities.

    Args:
        env: The simulation environment.
        position_range: ((min_offset_x, max_offset_x),
                         (min_offset_y, max_offset_y),
                         (min_offset_z, max_offset_z)).
                        These offsets are added to the base offset for randomness.
                        If None, no extra randomization is applied.
    """
    # print("reset Ball!!!!!")

    # Retrieve the kicking leg's toe position (shape: [N, 3])
    toe_pos = env.scene["kicking_leg_frame"].data.target_pos_w[..., 0, :]
    
    # Define a base offset, e.g., place the ball 0.1 m in front of the toe.
    base_offset = torch.tensor([0.0, 0, 0], device=toe_pos.device).unsqueeze(0)  # shape: [1, 3]
    position_range = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    # print("!!!! Randomized ball pos is run. Toe position:", toe_pos)
    
    # Determine random offset if position_range is provided.
    if position_range is not None:
        # position_range: ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        min_offsets = torch.tensor(
            [position_range[0][0], position_range[1][0], position_range[2][0]], device=toe_pos.device
        )
        max_offsets = torch.tensor(
            [position_range[0][1], position_range[1][1], position_range[2][1]], device=toe_pos.device
        )
        random_offsets = torch.rand(toe_pos.shape, device=toe_pos.device) * (max_offsets - min_offsets) + min_offsets
    else:
        random_offsets = torch.zeros_like(toe_pos)

    # print("toe_position", toe_pos)
    
    # Compute the new ball position.
    new_pos = toe_pos + base_offset + random_offsets  # shape: [N, 3]
        
    # Set the ball's linear velocity to zero.
    new_lin_vel = torch.zeros_like(new_pos)  # shape: [N, 3]
    
    # Set the ball's angular velocity to zero.
    new_ang_vel = torch.zeros_like(new_pos)  # shape: [N, 3]
    
    # Set the ball's orientation (quaternion) to the identity (no rotation).
    # (w, x, y, z) format.
    default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=toe_pos.device).unsqueeze(0)  # shape: [1, 4]
    default_quat = default_quat.expand(new_pos.shape[0], -1)  # shape: [N, 4]
    
    # Construct the full state vector for the ball: 
    # [position (3), quaternion (4), linear velocity (3), angular velocity (3)] => [N, 13]
    new_state = torch.cat([new_pos, default_quat, new_lin_vel, new_ang_vel], dim=-1)
    
    # Write the new state into simulation using the available methods.
    # The first 7 columns (position and quaternion) form the root pose.
    # The last 6 columns (linear and angular velocities) form the root velocity.
    # print("Old ball position:", env.scene["ball"].data.root_pos_w)
    # print("New ball position:", new_pos)

    ball_asset = env.scene["ball"]
    ball_asset.write_root_pose_to_sim(new_state[:, :7])
    ball_asset.write_root_velocity_to_sim(new_state[:, 7:])
    # print("Updated ball position:", env.scene["ball"].data.root_pos_w)

    


