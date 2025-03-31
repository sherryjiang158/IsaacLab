from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def approach_ball(env):
    """
    Reward for approaching the ball, with distance cap and max reward on kick event.

    Args:
        distance_cap: Maximum distance considered in reward calculation.
        ball_speed_threshold: If ball speed exceeds this, reward is set to max_reward.
        max_reward: The maximum reward value if ball is moving fast (i.e., kicked).
    """
    distance_threshold = 0.05  
    ball_speed_threshold = 0.3

    # Get the toe position
    toe_pos = env.scene["kicking_leg_frame"].data.target_pos_w[..., 0, :]
    # Get the ball position
    ball_pos = env.scene["ball"].data.root_pos_w
    # Compute Euclidean distance
    distance = torch.norm(ball_pos - toe_pos, dim=-1, p=2)

    # Compute base reward using capped distance
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)

    # Double the reward if within a close range
    reward = torch.where(distance <= distance_threshold, 2 * reward, reward)

    # Override reward if ball is moving (i.e., kicked)
    ball_vel = env.scene["ball"].data.root_lin_vel_w  # shape: [N, 3]
    ball_speed = torch.norm(ball_vel, dim=-1)
    reward = torch.where(ball_speed > ball_speed_threshold, 2, reward)

    return reward



def ball_displacement_reward(env):
    ball_data = env.scene["ball"].data
    current_pos = ball_data.root_state_w[:, :3]
    env_origins = env.scene.env_origins

    displacement = current_pos[:, 0] - env_origins[:, 0]
    print("displacement", displacement)
    max_displacement_reward = 4.0
    reward = torch.where(displacement > max_displacement_reward, max_displacement_reward, displacement)
    return reward


def kick_ball_velocity_reward(env):
    """
    Reward for kicking the ball effectively, based on the ball's velocity.
    
    This reward uses a multi-stage bonus approach:
      - The basic reward is the ball's speed (i.e. the Euclidean norm of its velocity).
      - An additional bonus is added if the speed exceeds a low threshold.
      - A further bonus is added if the speed exceeds a high threshold.
    
    Explanation:
      - Retrieves the ball's velocity vector.
      - Computes its magnitude (speed).
      - Adds bonus rewards in a piecewise fashion for higher speeds,
        encouraging the agent to produce a powerful kick.
    """
    ball_data = env.scene["ball"].data
    
    ball_vel = ball_data.root_state_w[:, 7:10]  # assumed shape [N, 3]
    # ball_vel_xy = ball_vel[:, :2]

    # desired_direction =   # shape: [N, 2]

    # Project the ball's xy velocity onto the robot's forward direction.
    # projected_speed = (ball_vel_xy * desired_direction).sum(dim=-1)  # dot product, shape: [N]
    projected_speed = ball_vel[:, 0] # just the velocity in x directions
    
    low_threshold = 0.4 # we can config this differently if we want to
    max_speed_reward = 1.0

    
    # Bonus of 0.5 if the ball speed is above low_threshold,
    # additional bonus of 0.5 if above high_threshold.
    bonus_low = torch.where(projected_speed > low_threshold, 0.2, torch.tensor(0.0, device=projected_speed.device))
    reward = torch.where(projected_speed > max_speed_reward, max_speed_reward, projected_speed)
    reward = reward + bonus_low
    # print("ball velocity reward shape", reward.shape)
    
    return reward


# def air_time_reward(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     # current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     reward = torch.clip(current_air_time, -mode_time, mode_time)

#     # Override reward if ball is moving (i.e., kicked)
#     ball_speed_threshold = 0.3
#     ball_vel = env.scene["ball"].data.root_lin_vel_w  # shape: [N, 3]
#     ball_speed = torch.norm(ball_vel, dim=-1)
#     reward = torch.where(ball_speed > ball_speed_threshold, mode_time, reward)

#     return torch.sum(reward, dim=1)

##
# Regularization Penalties
##


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


def joint_position_penalty_kick(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position error for the kicking task, where the robot's base remains static."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute the joint position error
    joint_error = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
    # Apply a constant scaling factor if needed
    return 5 * joint_error

def joint_position_penalty_kick(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position error for the kicking task, where the robot's base remains static."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Compute the joint position error
    joint_error = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
    # Apply a constant scaling factor if needed
    return 5 * joint_error



def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)


## omit this first? Since similar information would be captured by base orientation penalty?
## actually cannot omit that...
def support_feet_leave_ground_penalty(env):
    """
    Penalizes when support feet lose contact with the ground.
    
    This version uses separate contact sensor data for each support foot:
      - "contact_forces_fl" for the front-left foot,
      - "contact_forces_hl" for the hind-left foot, and
      - "contact_forces_hr" for the hind-right foot.
      
    For each sensor, if any of its measured force values exceeds a small threshold, 
    that foot is considered to be in contact (1), otherwise not (0).
    
    The penalty is computed as:
        penalty = - (total_feet - number_of_feet_in_contact)
    so that if all feet are in contact the penalty is 0, and if one or more are missing, 
    a negative penalty is applied.
    """
    sensor_names = ["contact_forces_fl", "contact_forces_hl", "contact_forces_hr"]
    threshold = 1e-6  # Force threshold to consider a foot in contact
    contacts = []
    
    for sensor_name in sensor_names:
        sensor_data = env.scene[sensor_name].data
        forces = sensor_data.net_forces_w  # Expected shape: [N, [[[[1.6198e-05, 8.1023e-05, 6.4741e+01]]]]]
        # If the sensor returns a multi-dimensional tensor, check if any force exceeds the threshold.
        if forces.ndim > 1:
            contact_indicator = (forces > threshold).any(dim=-1).float()
        else:
            contact_indicator = (forces > threshold).float()
        contacts.append(contact_indicator)
        # print("contact forces", forces)
    # print(contacts)
    # Stack contact indicators; shape: [N, num_sensors]
    contacts_tensor = torch.stack(contacts, dim=-1)
    # Count the number of feet in contact for each environment
    num_in_contact = contacts_tensor.sum(dim=-1)
    total_feet = float(len(sensor_names))
    missing_feet = total_feet - num_in_contact
    # The penalty is the negative number of missing feet
    penalty = missing_feet
    penalty = penalty.squeeze(-1)


    # print("supporting foot penalty:", penalty)
    # print("penalty shape", penalty.shape)

    return penalty

def root_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg 
) -> torch.Tensor:
    """Essentially for robot fall. Large negative reward when the robot's root height falls below a threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    minimum_height = 0.11

    root_height = asset.data.root_pos_w[:, 2]

    # Apply penalty where termination condition is met
    penalty = torch.where(
        root_height < minimum_height,
        torch.tensor(1.0, device=root_height.device),
        torch.tensor(0.0, device=root_height.device)
    )
    return penalty