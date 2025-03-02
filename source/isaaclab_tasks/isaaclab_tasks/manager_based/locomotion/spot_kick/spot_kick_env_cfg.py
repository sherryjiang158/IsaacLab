# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.spot_kick.mdp as mdp

from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the kicking scene."""
    
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # Robot configuration
    robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Rigid Object to create a ball
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.45),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    contact_forces_ball = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr_foot",  # Kicking foot
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],  # Only detect ball contact
    )

    contact_forces_support = ContactSensorCfg(
        prim_path=[
            "{ENV_REGEX_NS}/Robot/fl_foot",
            "{ENV_REGEX_NS}/Robot/hl_foot",
            "{ENV_REGEX_NS}/Robot/hr_foot"
        ],  # Support feet
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Ground"],  
        # No need to specify because if foot is in the air, there will be no contact force at all.
    )


# ##
# # MDP settings
# ##


# @configclass
# class ActionsCfg:
#     """Action specifications for the kicking MDP."""
    
#     joint_pos = mdp.JointPositionActionCfg(
#         asset_name="robot",
#         joint_names=[".*"],
#         scale=0.5,
#         use_default_offset=True
#     )

# @configclass
# class ObservationsCfg:
#     """Observation specifications for the kicking MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         # Robot state
#         base_lin_vel = ObsTerm(
#             func=mdp.base_lin_vel,
#             noise=Unoise(n_min=-0.1, n_max=0.1)
#         )
#         base_ang_vel = ObsTerm(
#             func=mdp.base_ang_vel,
#             noise=Unoise(n_min=-0.2, n_max=0.2)
#         )
#         projected_gravity = ObsTerm(
#             func=mdp.projected_gravity,
#             noise=Unoise(n_min=-0.05, n_max=0.05)
#         )
#         joint_pos = ObsTerm(
#             func=mdp.joint_pos_rel,
#             noise=Unoise(n_min=-0.01, n_max=0.01)
#         )
#         joint_vel = ObsTerm(
#             func=mdp.joint_vel_rel,
#             noise=Unoise(n_min=-1.5, n_max=1.5)
#         )
        
#         # Ball state
#         ball_relative_pos = ObsTerm(
#             func=mdp.ball_relative_position,
#             params={"asset_cfg": SceneEntityCfg("ball")},
#             noise=Unoise(n_min=-0.05, n_max=0.05)
#         )
#         ball_velocity = ObsTerm(
#             func=mdp.ball_velocity,
#             params={"asset_cfg": SceneEntityCfg("ball")},
#             noise=Unoise(n_min=-0.1, n_max=0.1)
#         )
        
#         # Last action
#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     policy: PolicyCfg = PolicyCfg()

# @configclass
# class EventCfg:
#     """Configuration for events."""

#     # Reset events
#     reset_robot_joints = EventTerm(
#         func=mdp.reset_joints_around_default,
#         mode="reset",
#         params={
#             "position_range": (-0.2, 0.2),
#             "velocity_range": (-0.5, 0.5),
#             "asset_cfg": SceneEntityCfg("robot"),
#         },
#     )

#     reset_ball = EventTerm(
#         func=mdp.reset_ball_position,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("ball"),
#             "position_range": {"x": (0.1, 0.3), "y": (-0.1, 0.1)},
#         },
#     )

# @configclass
# class RewardsCfg:
#     """Reward terms for the kicking MDP."""

#     # Kicking rewards
#     ball_forward_velocity = RewTerm(
#         func=mdp.ball_forward_velocity_reward,
#         weight=2.0,
#         params={"asset_cfg": SceneEntityCfg("ball")}
#     )
    
#     support_feet_stability = RewTerm(
#         func=mdp.support_feet_stability,
#         weight=1.0,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "sensor_cfg": SceneEntityCfg("contact_forces"),
#         }
#     )
    
#     # Penalties
#     joint_acceleration = RewTerm(
#         func=mdp.joint_acceleration_penalty,
#         weight=-0.1,
#         params={"asset_cfg": SceneEntityCfg("robot")}
#     )
    
#     base_orientation = RewTerm(
#         func=mdp.base_orientation_penalty,
#         weight=-1.0,
#         params={"asset_cfg": SceneEntityCfg("robot")}
#     )

#     support_feet_ground_penalty = RewTerm(
#         func=mdp.support_feet_leave_ground_penalty,
#         weight=5.0,  # High weight to strongly discourage lifting support feet
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "sensor_cfg": SceneEntityCfg("contact_forces"),
#             "penalty_scale": 5.0
#         }
#     )

# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)
#     robot_fall = DoneTerm(
#         func=mdp.robot_fall,
#         params={"asset_cfg": SceneEntityCfg("robot")}
#     )
#     successful_kick = DoneTerm(
#         func=mdp.successful_kick,
#         params={
#             "asset_cfg": SceneEntityCfg("ball"),
#             "min_velocity": 1.0
#         }
#     )

@configclass
class SpotKickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion kicking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    
    # # Basic settings
    # observations: ObservationsCfg = ObservationsCfg()
    # actions: ActionsCfg = ActionsCfg()
    
    # # MDP settings
    # rewards: RewardsCfg = RewardsCfg()
    # terminations: TerminationsCfg = TerminationsCfg()
    # events: EventCfg = EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
    #     """Post initialization."""
    #     # General settings
    #     self.decimation = 4
    #     self.episode_length_s = 4.0  # Shorter episodes for kicking
        
    #     # Simulation settings
    #     self.sim.dt = 0.005
    #     self.sim.render_interval = self.decimation
    #     self.sim.disable_contact_processing = False
        
    #     # Update sensor periods
    #     if self.scene.contact_forces is not None:
    #         self.scene.contact_forces.update_period = self.sim.dt


class SpotKickEnvCfg_PLAY(SpotKickEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False