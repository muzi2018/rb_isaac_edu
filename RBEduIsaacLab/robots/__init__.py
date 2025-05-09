import gymnasium as gym

# from . import agents
# import agents

##
# Register Gym environments.
##

gym.register(
    id="RoadBalance-Pendulum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pendulum_manager_based_env:CartpoleRLEnvCfg",
        "sb3_cfg_entry_point": f"{__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="RoadBalance-Pendulum-Direct-v0",
    entry_point=f"{__name__}.pendulum_direct_rl_env:CartpoleDirectRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pendulum_direct_rl_env:CartpoleEnvCfg",
        "sb3_cfg_entry_point": f"{__name__}:sb3_ppo_cfg.yaml",
    },
)

if __name__ == "__main__":
    print(f"{__name__}")