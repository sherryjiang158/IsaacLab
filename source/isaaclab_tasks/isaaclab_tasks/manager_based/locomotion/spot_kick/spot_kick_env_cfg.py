# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import torch
import random

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
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg


from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.spot_kick.mdp as mdp

from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


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

    #dx, dy, dz = random.uniform(0, 0.05), random.uniform(0, 0.05), random.uniform(0, 0.05)
    
    #init_ball_position = (0.1 + dx, 0.0 + dy, 0.0 + dz)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

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
        init_state=RigidObjectCfg.InitialStateCfg(), #pos omitted for initial position
    )

    contact_forces_ball = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr_foot",  # Kicking foot
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],  # Only detect ball contact
    )

    # contact_forces_support = ContactSensorCfg(
    #     prim_path=[
    #         "{ENV_REGEX_NS}/Robot/fl_foot",
    #         "{ENV_REGEX_NS}/Robot/hl_foot",
    #         "{ENV_REGEX_NS}/Robot/hr_foot"
    #     ],  # Support feet
    #     update_period=0.0,
    #     history_length=3,
    #     debug_vis=True,
    #     # filter_prim_paths_expr=["{ENV_REGEX_NS}/Ground"],  
    #     # No need to specify because if foot is in the air, there will be no contact force at all.
    # )
    contact_forces_fl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fl_foot",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        # optionally add filter_prim_paths_expr if needed
    )
    contact_forces_hr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hr_foot",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )
    contact_forces_hl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hl_foot",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )


    ball_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",  # The path where the ball is defined
        debug_vis=True,  # Enable visualization for debugging
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BallFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Sphere",  # Reference to the ball's geometry
                name="ball_center", 
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),  # Zero offset means the frame is at the ball's origin
                    rot=(1.0, 0.0, 0.0, 0.0)  # No rotation offset (identity quaternion)
                ),
            ),
        ],
    )

    kicking_leg_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr_foot",  # Base frame for the front right foot
        debug_vis=True, 
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/KickingLegFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/fr_foot",  # Using the foot as the reference
                name="toe",  
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),  # !!! may need adjest to find best contact point
                    rot=(1.0, 0.0, 0.0, 0.0)  # !!!!!! Need to further adjust
                ),
            ),
        ],
    )




# ##
# # MDP settings
# ##


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
        # Three terms that may be useful for keeping the balance
        
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
        ball_pos = ObsTerm(
            func=mdp.ball_position,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        ball_velocity = ObsTerm(
            func=mdp.ball_velocity,
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        # Leg to Ball Distance
        rel_ball_leg_distance = ObsTerm(func=mdp.rel_ball_leg_position)
        
        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # Randomize the ball's physical material properties (e.g., friction and bounciness)
    ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "static_friction_range": (0.1, 0.3),
            "dynamic_friction_range": (0.1, 0.3),
            "restitution_range": (0.8, 1.0),
            "num_buckets": 16,
        },
    )
   

    # Reset events

    # Reset the entire scene to the default state at the beginning of each episode
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # Reset the robot's joints with a small random offset 
    # to avoid always starting at exactly the same configuration
    # so that we have better robustness

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
     # Randomize the ball's starting position for additional variability
    randomize_ball_position = EventTerm(
        func=mdp.randomize_ball_position,
        mode="reset",
        # params={
        #     "position_range": ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0)), # !!!! May need to adjust based on code running situation
        # },
    )

    
@configclass
class RewardsCfg:
    """Reward terms for the kicking MDP."""

    air_time = RewTerm(
        func=mdp.air_time_reward,
        weight=0.5,
        params={
            "mode_time": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="fr_foot"),
        },
    )

    # 1. Approach the ball: 
    # Encourage the kicking leg's toe to get close to the ball.
    approach_ball = RewTerm(
        func=mdp.approach_ball,
        weight=0.5,
    )
    ball_displacement = RewTerm(
        func=mdp.ball_displacement_reward,
        weight = 5.0
    )
    
    # # 2. Alignment of kicking leg: Reward for aligning the kicking leg (toe) properly with the ball.
    # align_kicking_leg = RewTerm(
    #     func=mdp.align_kicking_leg,
    #     weight=0.5
    # )
    
    # 3. Kick impact: Reward for transferring momentum to the ball 
    # (measured by ball velocity post-impact).
    kick_ball_velocity = RewTerm(
        func=mdp.kick_ball_velocity_reward,
        weight=5.0
    )
    
    # Let's not work on this yet... right now, just focus on get the ball rolling.
    # Target bonus reward if the ball reaches a desired area.
    # target_ball_bonus = RewTerm(
    #     func=mdp.target_ball_bonus,
    #     weight=7.5,
    #     params={"target_pos": (1.0, 0.0, 0.0), "tolerance": 0.2}
    # )
    
    # 4. Penalize rapid changes in actions to promote smoother control. (smaller)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-3
    )
    
    # 5. Penalize excessive joint velocities to ensure stable and controlled motion. (smaller)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4
    )
    
    # Penalties

    support_feet_leave_ground_penalty = RewTerm(
        func=mdp.support_feet_leave_ground_penalty,
        weight=-5.0,  # High weight to strongly discourage lifting support feet
    )

    # -- penalties
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.7)

    base_motion = RewTerm(
        func=mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # base_displacement = RewTerm(
    # )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty_kick,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # body_contact = DoneTerm(
    #     func=mdp.illegal_contact_kick,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]), "threshold": 1.0},
    # )
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.5},
    )


@configclass
class SpotKickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion kicking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
    #     """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 2.0  # Shorter episodes for kicking
        
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        
        # Update sensor periods
        # if self.scene.contact_forces_support is not None:
        #     self.scene.contact_forces_support.update_period = self.sim.dt
        if self.scene.contact_forces_fl is not None:
            self.scene.contact_forces_fl.update_period = self.sim.dt
        if self.scene.contact_forces_hl is not None:
            self.scene.contact_forces_hl.update_period = self.sim.dt        
        if self.scene.contact_forces_hr is not None:
            self.scene.contact_forces_hr.update_period = self.sim.dt


class SpotKickEnvCfg_PLAY(SpotKickEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # # disable randomization for play
        # self.observations.policy.enable_corruption = False