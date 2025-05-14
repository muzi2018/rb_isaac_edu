# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32
    or
    isaaclab -p my_create_cartpole_base_env.py --num_envs 32 --headless
    isaaclab -p my_create_cartpole_base_env_pid.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np

from isaaclab.envs import ManagerBasedEnv

# Environment configuration can be found from below code.
from robots.pendulum_manager_based_env import CartpoleEnvCfg

theta_des = np.pi
kp = 200
kd = 2 * np.sqrt(kp)
# pure control 30 ok / 25 x
min_torque, max_torque = -25, 25

def controller(obs):
    theta, omega = obs[0]

    # Ensure tensors are on CPU before converting to numpy
    torque_value = -kp * (theta.cpu().numpy() - theta_des) - kd * omega.cpu().numpy()

    # Clip torque value to specified limits
    torque_value = np.clip(torque_value, min_torque, max_torque)

    # Wrap in 2D NumPy array to get shape [1, 1]
    torque = torch.from_numpy(np.array([[torque_value]], dtype=np.float32)).to(theta.device)

    return torque

def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    obs = None
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            if obs is not None:
                joint_efforts = controller(obs["policy"])
                print(f"joint_efforts: {joint_efforts} {joint_efforts.size()} {type(joint_efforts)}")

                # step the environment
                obs, extra = env.step(joint_efforts)
            else:
                random_efforts = torch.randn_like(env.action_manager.action)
                print(f"random_efforts: {random_efforts} {random_efforts.size()} {type(random_efforts)}")

                # step the environment
                obs, extra = env.step(random_efforts)
                        
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][0].item())
            print("[Env 0]: Pole vel: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()