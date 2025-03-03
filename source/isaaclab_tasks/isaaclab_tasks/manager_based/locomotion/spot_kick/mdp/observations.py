from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def rel_ball_leg_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compute the relative distance between the ball and the kicking leg's toe.
    
    This returns a vector from the toe frame (kicking_leg_frame) to the ball's root position.
    """
    # Get the frame data for the kicking leg (toe)
    leg_tf_data = env.scene["kicking_leg_frame"].data
    # Get the ball's data
    ball_data = env.scene["ball"].data

    # Compute the vector difference: ball position minus toe position.
    # [# envs, which target frame, : all coordinates]
    return ball_data.root_pos_w - leg_tf_data.target_pos_w[..., 0, :]


