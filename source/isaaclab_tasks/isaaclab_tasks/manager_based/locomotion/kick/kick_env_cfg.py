# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.kick.mdp as mdp

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the kicking scene."""
    
    # Simple flat ground for kicking
    ground = sim_utils.RigidBodyCfg(
        prim_path="/World/ground",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    
    # Robot configuration
    robot: ArticulationCfg = MISSING
    
    # Ball configuration
    ball = sim_utils.RigidBodyCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        mass=0.45,  # Soccer ball mass (kg)
        radius=0.11,  # Soccer ball radius (m)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.7,
        ),
    )
    
    # Contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )

@configclass
class ActionsCfg:
    """Action specifications for the kicking MDP."""
    
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the kicking MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        
        # Ball state
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
        
        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # Reset events
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_ball_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "position_range": {"x": (0.1, 0.3), "y": (-0.1, 0.1)},
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the kicking MDP."""

    # Kicking rewards
    ball_forward_velocity = RewTerm(
        func=mdp.ball_forward_velocity_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("ball")}
    )
    
    support_feet_stability = RewTerm(
        func=mdp.support_feet_stability,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        }
    )
    
    # Penalties
    joint_acceleration = RewTerm(
        func=mdp.joint_acceleration_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    support_feet_ground_penalty = RewTerm(
        func=mdp.support_feet_leave_ground_penalty,
        weight=5.0,  # High weight to strongly discourage lifting support feet
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "penalty_scale": 5.0
        }
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(
        func=mdp.robot_fall,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    successful_kick = DoneTerm(
        func=mdp.successful_kick,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "min_velocity": 1.0
        }
    )

@configclass
class LocomotionKickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion kicking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 4.0  # Shorter episodes for kicking
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        
        # Update sensor periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt