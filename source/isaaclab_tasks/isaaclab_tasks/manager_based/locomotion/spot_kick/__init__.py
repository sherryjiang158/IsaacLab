# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Kick-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_kick_env_cfg:SpotKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotKickPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_kick_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Kick-Spot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_kick_env_cfg:SpotKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotKickPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_kick_ppo_cfg.yaml",
    },
)