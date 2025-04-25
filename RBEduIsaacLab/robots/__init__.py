# from .simulation import Simulator
# from .pendulum_plant import PendulumPlant
# from .gym_environment import SimplePendulumEnv

# from gymnasium.envs.registration import register

# register(
#     id="gymnasium_env/SimplePendulum-v0",
#     entry_point=SimplePendulumEnv,
#     kwargs={
#         "simulator": None
#     }
# )


import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="RoadBalance-Pendulum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pendulum_manager_based_env:CartpoleRLEnvCfg",
    },
)