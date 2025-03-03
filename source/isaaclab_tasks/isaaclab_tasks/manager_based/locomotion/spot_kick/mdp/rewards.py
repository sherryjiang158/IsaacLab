from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

def approach_ball(env, params):
    """
    Reward for approaching the ball using an inverse-square law.
    
    This reward is computed based on the Euclidean distance between the kicking leg's toe
    and the ball. It uses a piecewise scaling:
      - The reward is given by (1/(1 + distance^2))^2.
      - If the toe is within a given threshold, the reward is doubled.
    
    Parameters in params:
      - "threshold": distance below which the reward is boosted (default: 0.1 meters)
      
    Explanation:
      - Retrieves the toe position from the kicking_leg_frame.
      - Retrieves the ball's position from its data.
      - Computes the Euclidean distance.
      - Applies the inverse-square law and squares it for sharper decay.
      - Uses torch.where to double the reward when within the threshold.
    """
    threshold = params.get("threshold", 0.1)
    # Get the toe's world position (assumes shape [N, 3])
    toe_pos = env.scene["kicking_leg_frame"].data.target_pos_w[..., 0, :]
    # Get the ball's world position (assumed to be in root_pos_w)
    ball_pos = env.scene["ball"].data.root_pos_w
    # Compute Euclidean distance between toe and ball
    distance = torch.norm(ball_pos - toe_pos, dim=-1, p=2)
    
    # Inverse-square law reward, squared for sharper decay with distance
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    # If the toe is within the threshold, double the reward
    reward = torch.where(distance <= threshold, 2 * reward, reward)
    return reward


def kick_ball_velocity(env, params):
    """
    Reward for kicking the ball effectively, based on the ball's velocity.
    
    This reward uses a multi-stage bonus approach:
      - The basic reward is the ball's speed (i.e. the Euclidean norm of its velocity).
      - An additional bonus is added if the speed exceeds a low threshold.
      - A further bonus is added if the speed exceeds a high threshold.
    
    Parameters in params:
      - "low_threshold": speed threshold for a bonus (default: 0.5)
      - "high_threshold": higher speed threshold for an extra bonus (default: 1.0)
    
    Explanation:
      - Retrieves the ball's velocity vector.
      - Computes its magnitude (speed).
      - Adds bonus rewards in a piecewise fashion for higher speeds,
        encouraging the agent to produce a powerful kick.
    """
    ball_data = env.scene["ball"].data
    ball_vel = ball_data.velocity  # assumed shape [N, 3]
    speed = torch.norm(ball_vel, dim=-1)
    
    low_threshold = params.get("low_threshold", 0.5) # we can config this differently if we want to
    high_threshold = params.get("high_threshold", 1.0)
    
    # Bonus of 0.5 if the ball speed is above low_threshold,
    # additional bonus of 0.5 if above high_threshold.
    bonus_low = torch.where(speed > low_threshold, 0.5, torch.tensor(0.0, device=speed.device))
    bonus_high = torch.where(speed > high_threshold, 0.5, torch.tensor(0.0, device=speed.device))
    
    return speed + bonus_low + bonus_high


def action_rate_l2(env, params):
    """
    Penalizes rapid changes in actions (L2 norm of the difference between current and last actions).
    
    Explanation:
      - Assumes env.current_action and env.last_action are tensors of shape [N, action_dim].
      - Computes the squared L2 norm of their difference.
      - Returns a negative penalty value (so larger changes yield larger negative rewards).
    """
    current_action = env.current_action  # shape: [N, action_dim]
    last_action = env.last_action          # shape: [N, action_dim]
    diff = current_action - last_action
    penalty = torch.norm(diff, dim=-1) ** 2
    return -penalty


def joint_vel_l2(env, params):
    """
    Penalizes high joint velocities.
    
    Explanation:
      - Retrieves the robot's joint velocities from its data (assumed shape [N, num_joints]).
      - Computes the squared L2 norm across joints.
      - Returns a negative penalty to discourage unnecessarily fast or erratic motions.
    """
    robot_data = env.scene["robot"].data
    joint_vel = robot_data.joint_vel  # shape: [N, num_joints]
    penalty = torch.norm(joint_vel, dim=-1) ** 2
    return -penalty


def base_orientation_penalty(env, params):
    """
    Penalizes deviations of the robot's base orientation from the desired upright orientation.
    
    Explanation:
      - Retrieves the robot's base orientation as a quaternion (shape [N, 4]).
      - Compares it with the desired upright quaternion (0, 0, 0, 1),
        which represents no rotation relative to the world frame (i.e. perfectly upright).
      - The similarity is measured via the dot product between the two quaternions.
      - Returns a penalty that increases as the base deviates from upright.
    """
    robot_data = env.scene["robot"].data
    base_quat = robot_data.base_rot_w  # shape: [N, 4]
    # Desired upright orientation: (0, 0, 0, 1) in quaternion (x, y, z, w) format.
    desired_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base_quat.device).unsqueeze(0)
    # Compute the dot product as a similarity measure (1 means perfect alignment)
    dot = torch.abs(torch.sum(base_quat * desired_quat, dim=-1)) # Notice that the dot product is for alignment
    penalty = 1.0 - dot  # Zero penalty when perfectly aligned; up to 1 when completely misaligned.
    return -penalty


def support_feet_leave_ground_penalty(env, params):
    """
    Penalizes when the support feet lose contact with the ground.
    
    Explanation:
      - Uses contact sensor data from support feet (e.g., defined under "contact_forces_support").
      - Checks if each foot's measured force exceeds a small threshold, indicating contact.
      - Computes the fraction of support feet in contact.
      - Returns a negative penalty that increases as fewer feet are in contact.
    """
    sensor_data = env.scene["contact_forces_support"].data
    forces = sensor_data.force  # assumed shape: [N, num_feet]
    threshold = 1e-5  # threshold to consider a foot in contact, !!!! we may need to adjust this 
    contact = (forces > threshold).float()  # 1 if in contact, 0 otherwise
    contact_fraction = contact.sum(dim=-1) / forces.shape[-1]
    penalty = 1.0 - contact_fraction  # 0 if all feet are in contact; 1 if none.
    return -penalty
