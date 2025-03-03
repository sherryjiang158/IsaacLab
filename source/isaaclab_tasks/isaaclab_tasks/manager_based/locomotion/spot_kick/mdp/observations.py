from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def rel_ball_leg_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute the relative distance between the ball and the kicking leg's toe.
    
    This returns a vector from the toe frame (kicking_leg_frame) to the ball's root position.
    So it's technically relative position.
    """
    # Get the frame data for the kicking leg (toe)
    leg_tf_data = env.scene["kicking_leg_frame"].data
    # Get the ball's data
    ball_data = env.scene["ball"].data

    # Compute the vector difference: ball position minus toe position.
    # [# envs, which target frame, : all coordinates]
    return ball_data.root_pos_w - leg_tf_data.target_pos_w[..., 0, :]

def ball_velocity(env):
    """
    Returns the ball's velocity.
    Since it's rigid object. Maybe we should see RigidObjectData:
    https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData
    """
    ball_data = env.scene["ball"].data
    print("print: root_state_w", ball_data.root_state_w)
    vel = ball_data.root_state_w[2] # root_state_w Root state [pos, quat, lin_vel, ang_vel] in simulation world frame.
    return vel 

def ball_position(env):
    """
    Returns the ball's position.
    
    Explanation:
      - For simplicity, we return the ball's world position (root_pos_w) to see if it works????
    """
    ball_data = env.scene["ball"].data
    return ball_data.root_pos_w  # shape: [N, 3]
