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
    isaaclab -p my_create_cartpole_base_env_ilqr_pid.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import control
import numpy as np

from functools import partial
from isaaclab.envs import ManagerBasedEnv

# Environment configuration can be found from below code.
from robots.pendulum_manager_based_env import CartpoleEnvCfg
from model_based_control import (
    PIDController,
    iLQR_Calculator,
    prepare_empty_data_dict,
    plot_trajectory,
    save_trajectory,
    plot_ilqr_trace,
    pendulum_discrete_dynamics_rungekutta,
    pendulum_swingup_stage_cost,
    pendulum_swingup_final_cost,
)

class Parameters():

    def __init__(self):
        # System parameters
        self.m = 13.5  # mass of pendulum
        self.l = 0.19  # length of pendulum
        self.c = 0.0  # friction coefficient
        self.g = 9.81  # gravity
        self.I = self.m * self.l**2  # moment of inertia
        self.b = 0.1  # damping coefficient
        self.n_x = 2
        self.n_u = 1
        self.integrator = "runge_kutta"

        # swingup parameters
        self.dt = 0.01
        self.x0 = [0.0, 0.0]
        self.goal = [np.pi, 0.0]
        # 5 X / 10 OK / 7.5 O / 6.0 X
        self.torque_limit = 6.0

        # iLQR parameters
        self.total_timestamp = 1000
        self.max_iter = 100
        self.regu_init = 100
        # cost function weights
        self.sCu = 0.1
        self.sCp = 0.0
        self.sCv = 0.0
        self.sCen = 0.0
        self.fCp = 1000.0
        self.fCv = 1.0
        self.fCen = 0.0

        # Control parameters
        # TODO: FInal Error minimization
        self.Kp = 25.0
        self.Kd = 1.0
        self.Ki = 0.0


def controller(obs, pid_controller, torque_limit, counter):
    theta, omega = obs[0]
    
    # Convert torch tensors to numpy arrays using detach() to avoid circular imports
    theta_np = theta.detach().cpu().numpy()
    omega_np = omega.detach().cpu().numpy()

    _, _, torque = pid_controller.get_control_output(counter, theta_np, omega_np)

    torque = np.clip(torque, -torque_limit, torque_limit)

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

    # Trajectory Optimization
    iLQR = iLQR_Calculator(n_x=params.n_x, n_u=params.n_u)
    dyn_func = pendulum_discrete_dynamics_rungekutta

    # set dynamics
    dyn = partial(
        dyn_func, 
        dt=params.dt,
        m=params.m, 
        l=params.l, 
        b=params.b, 
        cf=params.c, 
        g=params.g, 
        inertia=params.I
    )
    iLQR.set_discrete_dynamics(dyn)

    # set costs
    s_cost_func = pendulum_swingup_stage_cost
    f_cost_func = pendulum_swingup_final_cost
    s_cost = partial(
        s_cost_func,
        goal=params.goal,
        Cu=params.sCu,
        Cp=params.sCp,
        Cv=params.sCv,
        Cen=params.sCen,
        m=params.m,
        l=params.l,
        b=params.b,
        cf=params.c,
        g=params.g
    )
    f_cost = partial(
        f_cost_func,
        goal=params.goal,
        Cp=params.fCp,
        Cv=params.fCv,
        Cen=params.fCen,
        m=params.m,
        l=params.l,
        b=params.b,
        cf=params.c,
        g=params.g
    )
    iLQR.set_stage_cost(s_cost)
    iLQR.set_final_cost(f_cost)

    iLQR.init_derivatives()
    iLQR.set_start(params.x0)

    # computation
    (x_trj, u_trj, cost_trace, regu_trace,
    redu_ratio_trace, redu_trace) = iLQR.run_ilqr(
        N=params.total_timestamp,
        init_u_trj=None,
        init_x_trj=None,
        max_iter=params.max_iter,
        regu_init=params.regu_init,
        break_cost_redu=1e-6
    )

    # preprocess results
    time = np.linspace(0, params.total_timestamp-1, params.total_timestamp)*params.dt
    TH = x_trj.T[0]
    THD = x_trj.T[1]

    data_dict = prepare_empty_data_dict(params.dt, params.total_timestamp*params.dt)
    data_dict["des_time"] = time
    data_dict["des_pos"] = TH
    data_dict["des_vel"] = THD
    data_dict["des_tau"] = np.append(u_trj.T[0], 0.0)

    # save results
    log_dir = "ilqr"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    csv_path = os.path.join(log_dir, "computed_trajectory.csv")
    save_trajectory(csv_path, data_dict)

    # # plot results
    # plot_trajectory(time, x_trj, data_dict["des_tau"], None, True)
    # plot_ilqr_trace(cost_trace, redu_ratio_trace, regu_trace)

    pid_controller = PIDController(
        data_dict=data_dict, 
        Kp=params.Kp, 
        Ki=params.Ki,
        Kd=params.Kd,
        use_feed_forward=True
    )
    pid_controller.set_goal(params.goal)

    # # simulate physics
    # count = 0
    # obs = None
    # while simulation_app.is_running():
    #     with torch.inference_mode():
    #         # reset
    #         if count % (params.total_timestamp + 20) == 0:
    #             count = 0
    #             env.reset()
    #             pid_controller.reset()
    #             print("-" * 80)
    #             print("[INFO]: Resetting environment...")
    #         # sample random actions
    #         if obs is not None:
    #             joint_efforts = controller(obs["policy"], pid_controller, params.torque_limit, count)
    #             print(f"count: {count} / joint_efforts: {joint_efforts} {joint_efforts.size()} {type(joint_efforts)}")

    #             # step the environment
    #             obs, extra = env.step(joint_efforts)
    #         else:
    #             random_efforts = torch.randn_like(env.action_manager.action)
    #             print(f"random_efforts: {random_efforts} {random_efforts.size()} {type(random_efforts)}")

    #             # step the environment
    #             obs, extra = env.step(random_efforts)
                        
    #         # print current orientation of pole
    #         print("[Env 0]: Pole joint: ", obs["policy"][0][0].item())
    #         print("[Env 0]: Pole vel: ", obs["policy"][0][1].item())
    #         # update counter
    #         count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()