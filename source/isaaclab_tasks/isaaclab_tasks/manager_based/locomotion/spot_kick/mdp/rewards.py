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
    Reward for approaching the ball using an inverse-square law.
    
    This reward is computed based on the Euclidean distance between the kicking leg's toe
    and the ball. It uses a piecewise scaling:
      - The reward is given by (1/(1 + distance^2))^2.
      - If the toe is within a given threshold, the reward is doubled.
      
    Explanation:
      - Retrieves the toe position from the kicking_leg_frame.
      - Retrieves the ball's position from its data.
      - Computes the Euclidean distance.
      - Applies the inverse-square law and squares it for sharper decay.
      - Uses torch.where to double the reward when within the threshold.
    """
    threshold = 0.1
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


def kick_ball_velocity(env):
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
    speed = torch.norm(ball_vel, dim=-1)
    
    low_threshold = 0.5 # we can config this differently if we want to
    high_threshold = 1.0
    
    # Bonus of 0.5 if the ball speed is above low_threshold,
    # additional bonus of 0.5 if above high_threshold.
    bonus_low = torch.where(speed > low_threshold, 0.5, torch.tensor(0.0, device=speed.device))
    bonus_high = torch.where(speed > high_threshold, 0.5, torch.tensor(0.0, device=speed.device))

    reward = speed + bonus_low + bonus_high
    print("reward shape", reward.shape)
    
    return reward


# def action_rate_l2(env, params):
#     """
#     Penalizes rapid changes in actions (L2 norm of the difference between current and last actions).
    
#     Explanation:
#       - Assumes env.current_action and env.last_action are tensors of shape [N, action_dim].
#       - Computes the squared L2 norm of their difference.
#       - Returns a negative penalty value (so larger changes yield larger negative rewards).
#     """
#     current_action = env.current_action  # shape: [N, action_dim]
#     last_action = env.last_action          # shape: [N, action_dim]
#     diff = current_action - last_action
#     penalty = torch.norm(diff, dim=-1) ** 2
#     return -penalty


# def joint_vel_l2(env, params):
#     """
#     Penalizes high joint velocities.
    
#     Explanation:
#       - Retrieves the robot's joint velocities from its data (assumed shape [N, num_joints]).
#       - Computes the squared L2 norm across joints.
#       - Returns a negative penalty to discourage unnecessarily fast or erratic motions.
#     """
#     robot_data = env.scene["robot"].data
#     joint_vel = robot_data.joint_vel  # shape: [N, num_joints]
#     penalty = torch.norm(joint_vel, dim=-1) ** 2
#     return -penalty



def base_orientation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    This is adapted from Spot walking example
    """
    # extract the used quantities (to enable type-hinting)
    robot_data = env.scene["robot"].data
    return torch.linalg.norm((robot_data.projected_gravity_b[:, :2]), dim=1)


# def support_feet_leave_ground_penalty(env, params):
#     """
#     Penalizes when the support feet lose contact with the ground.
    
#     Explanation:
#       - Uses contact sensor data from support feet (e.g., defined under "contact_forces_support").
#       - Checks if each foot's measured force exceeds a small threshold, indicating contact.
#       - Computes the fraction of support feet in contact.
#       - Returns a negative penalty that increases as fewer feet are in contact.
#     """
#     sensor_data = env.scene["contact_forces_support"].data
#     forces = sensor_data.force  # assumed shape: [N, num_feet]
#     threshold = 1e-5  # threshold to consider a foot in contact, !!!! we may need to adjust this 
#     contact = (forces > threshold).float()  # 1 if in contact, 0 otherwise
#     contact_fraction = contact.sum(dim=-1) / forces.shape[-1]
#     penalty = 1.0 - contact_fraction  # 0 if all feet are in contact; 1 if none.
#     return -penalty


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
        print("contact forces", forces)
    
    # Stack contact indicators; shape: [N, num_sensors]
    contacts_tensor = torch.stack(contacts, dim=-1)
    # Count the number of feet in contact for each environment
    num_in_contact = contacts_tensor.sum(dim=-1)
    total_feet = float(len(sensor_names))
    missing_feet = total_feet - num_in_contact
    # The penalty is the negative number of missing feet
    penalty = -missing_feet
    print("supporting foot penalty:", penalty)
    print("penalty shape", penalty.shape)
    return penalty

