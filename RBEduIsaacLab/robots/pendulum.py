import os
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

current_folder_path = os.path.dirname(os.path.abspath(__file__))
# my_usd_file_path = os.path.join(current_folder_path, '..', 'USD', 'pendulum.usd')
my_usd_file_path = os.path.join(current_folder_path, '..', '..', 'USD', 'pendulum.usd')
print(f"{my_usd_file_path=}")


PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=my_usd_file_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4), joint_pos={"continuous_joint": 0.0}
    ),
    actuators={
        "pendulum_actuator": ImplicitActuatorCfg(
            joint_names_expr=["continuous_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=100.0,
            # stiffness=5.0,
            stiffness=30.0,
            damping=30.0,
        ),
    },
)