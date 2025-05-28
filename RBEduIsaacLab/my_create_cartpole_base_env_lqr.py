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
    isaaclab -p my_create_cartpole_base_env_lqr.py --num_envs 1

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
import control
import numpy as np

from isaaclab.envs import ManagerBasedEnv

# Environment configuration can be found from below code.
from robots.pendulum_manager_based_env import CartpoleEnvCfg

class Parameters():

    def __init__(self):
        # System parameters
        self.m = 13.5  # mass of pendulum
        self.l = 0.19  # length of pendulum
        self.g = 9.81  # gravity
        self.I = self.m * self.l**2  # moment of inertia
        self.b = 0.1  # damping coefficient
        
        # Control parameters
        self.theta_des = np.pi

        # Torque limits
        self.min_torque = -25  # Fixed variable name
        self.max_torque = 25  # Fixed variable name


def linearize(goal, params):

    m, l, g, I, b = params.m, params.l, params.g, params.I, params.b

    A = np.array([
        [0, 1],
        [-m*g*l/I * np.cos(goal[0]), -b/I]
    ])
    B = np.array([[0, 1./I]]).T

    return A, B

def controller(obs, K, params):  # Added K and params as arguments
    theta, omega = obs[0]
    
    # Convert torch tensors to numpy arrays using detach() to avoid circular imports
    theta_np = theta.detach().cpu().numpy()
    omega_np = omega.detach().cpu().numpy()

    goal = [params.theta_des, 0.0]

    delta_pos = theta_np - goal[0]
    delta_pos_wrapped = (delta_pos + np.pi) % (2*np.pi) - np.pi
    delta_y = np.array([delta_pos_wrapped, omega_np - goal[1]])

    torque = float(-K.dot(delta_y)[0])

    torque = np.clip(torque, -params.max_torque, params.max_torque)  # Use params limits

    # Wrap in 2D NumPy array to get shape [1, 1]
    torque = torch.tensor([[torque]], dtype=torch.float32, device=theta.device)

    return torque

def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Initialize parameters
    params = Parameters()

    ### LQR
    # Linearize for linear control
    Q = np.diag([10, 1])
    R = np.array([[1]])
    goal = np.array([np.pi, 0])
    A_lin, B_lin = linearize(goal, params)
    # K, S, eigVals = lqr_scipy(A_lin, B_lin, Q, R)
    K, S, eigVals = control.lqr(A_lin, B_lin, Q, R)
    print(f"{K=} {eigVals=}")

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
                joint_efforts = controller(obs["policy"], K, params)
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