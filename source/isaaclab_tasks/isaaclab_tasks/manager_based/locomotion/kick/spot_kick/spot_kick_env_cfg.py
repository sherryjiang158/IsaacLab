# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.kick.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.kick.kick_env_cfg import LocomotionKickEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # All joints
        scale=0.2,
        use_default_offset=True
    )

@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state observations
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5)
        )

        # Ball state observations
        ball_relative_pos = ObsTerm(
            func=mdp.ball_relative_position,
            params={"asset_cfg": SceneEntityCfg("ball")},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        ball_velocity = ObsTerm(
            func=mdp.ball_velocity,
            params={"asset_cfg": SceneEntityCfg("ball")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        # Last actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Reset events
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),  # Smaller range for stable standing
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_ball_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "position_range": {
                "x": (0.2, 0.3),    # Ball in front of right foot
                "y": (-0.1, 0.1)    # Small lateral variation
            },
        },
    )

@configclass
class SpotRewardsCfg:
    """Reward specifications for the MDP."""

    # Task rewards
    ball_velocity = RewardTermCfg(
        func=mdp.ball_forward_velocity_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "target_velocity": 2.0,
            "std": 0.5
        }
    )

    kick_impact = RewardTermCfg(
        func=mdp.kick_impact_reward,
        weight=1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "foot_cfg": SceneEntityCfg("robot", body_names=["front_right_foot"]),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "optimal_force": 20.0,
            "force_std": 5.0
        }
    )

    # Stability rewards/penalties
    support_feet = RewardTermCfg(
        func=mdp.support_feet_stability,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "min_force": 20.0,
            "max_force": 100.0,
            "history_steps": 3
        }
    )

    support_feet_ground = RewardTermCfg(
        func=mdp.support_feet_leave_ground_penalty,
        weight=-5.0,  # High negative weight to strongly discourage lifting support feet
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "penalty_scale": 5.0
        }
    )

    # Motion penalties
    action_smoothness = RewardTermCfg(
        func=mdp.action_smoothness_penalty,
        weight=-0.1
    )

    base_motion = RewardTermCfg(
        func=mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    base_orientation = RewardTermCfg(
        func=mdp.base_orientation_penalty,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class SpotTerminationsCfg:
    """Termination specifications for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    robot_fall = DoneTerm(
        func=mdp.robot_fall,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    successful_kick = DoneTerm(
        func=mdp.successful_kick,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "min_velocity": 1.0,
            "max_height": 0.5
        }
    )

    ball_out = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "bounds": 3.0
        }
    )

@configclass
class SpotKickEnvCfg(LocomotionKickEnvCfg):
    """Configuration for Spot kicking environment."""

    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()

    # MDP settings
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer settings
    viewer = ViewerCfg(
        eye=(7.5, 7.5, 3.0),
        origin_type="world",
        env_index=0,
        asset_name="robot"
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        # Shorter episodes for kicking task
        self.episode_length_s = 4.0
        
        # Higher control frequency for precise kicking
        self.decimation = 2
        self.sim.dt = 0.002
        
        # Scene settings
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        
        # Robot configuration
        self.scene.robot = SPOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state={
                "pos": (0.0, 0.0, 0.0),  # Start at origin
                "rot": (1.0, 0.0, 0.0, 0.0),  # Identity quaternion
                "joint_pos": "default",  # Use default standing pose
                "joint_vel": (0.0,),  # Zero velocity
            }
        )