from omni.isaac.etri_env import add_object, add_bg, add_random_object_visual, add_random_object_physics
from omni.isaac.etri_env import add_physical_properties, add_friction_properties
from omni.isaac.etri_env import add_sphere_light, add_random_light

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.materials.physics_material import PhysicsMaterial
import omni.isaac.core.utils.prims as prims_utils

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema

# Franka Robot
from omni.isaac.custom_controller import CustomPickPlaceController
from omni.isaac.robo_data_collector import RoboDataCollector
from omni.isaac.custom_franka import Franka2F, Franka3F
from omni.isaac.franka import Franka

from omni.isaac.etri_env import MultiEnvRandomizer
from .nucleus_variables import *

import numpy as np
import omni
import carb


class SemiconductorRandomizerExample(MultiEnvRandomizer):

    def __init__(
            self,
            world,
            json_file_path=None,
            verbose=True,
        ) -> None:

        super().__init__(
            world, 
            json_file_path, 
            verbose
        )

        return


    def setup_scene(self):
        super().setup_scene()
        return


    def add_task(self, index, env_offset):
        self._task_type = self._task_config["task_type"]
        cur_robot = self._robots[index]

        if self._task_type == "pick_and_place":
            end_effector_initial_height = self._task_config["end_effector_initial_height"]

            controller = CustomPickPlaceController(
                name=f"pick_place_controller_{index}",
                gripper=cur_robot.gripper,
                robot_articulation=cur_robot,
                end_effector_initial_height=end_effector_initial_height,
                robot_type=self._robot_type,
            )

            self._picking_object_name = self._task_config["picking_object_name"][0]
            self._end_effector_offset = np.array(self._task_config["end_effector_offset"])
            self._object_offset = np.array(self._task_config["object_offset"])
            
            self._picking_offset = np.array(self._task_config["picking_offset"])
            self._picking_orientation = np.array(self._task_config["picking_orientation"])
            self._placing_position = np.array(self._task_config["placing_position"])
            self._placing_offset = np.array(self._task_config["placing_offset"])
            self._placing_orientation = np.array(self._task_config["placing_orientation"])

        self._controllers.append(controller)


    async def setup_post_load(self):
        await super().setup_post_load()
        return


    def forward(self):
        super().forward()

        for i, robot in enumerate(self._robots):

            # exception : franka normal에서 벌어지지 않는 경우 발생
            if self._save_count < 10:
                robot.gripper.open()

            picking_object_name = f"Env_{i}_{self._picking_object_name}"
            object_geom = self._world.scene.get_object(picking_object_name)
            object_position, _ = object_geom.get_world_pose()

            picking_position = object_position + self._object_offset
            
            env_offset = self._env_offsets[i]
            placing_position = env_offset + self._placing_position + self._object_offset
            print(f"env_offset: {env_offset}")
            print(f"self._placing_position: {self._placing_position}")
            print(f"self._object_offset: {self._object_offset}")

            # robot frame to world frame 
            picking_orientation_world = rot_matrix_to_quat(
                quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._picking_orientation)
            )
            placing_orientation_world = rot_matrix_to_quat(
                quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._placing_orientation)
            )

            actions = self._controllers[i].forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=robot.get_joint_positions(),
                picking_offset=self._picking_offset,
                placing_offset=self._placing_offset,
                end_effector_offset=self._end_effector_offset,
                picking_end_effector_orientation=picking_orientation_world,
                placing_end_effector_orientation=placing_orientation_world,
            )
            if self._controllers[i].is_done():
                print(f"[SemiconductorRandomizerExample] done picking and placing")
            else:
                robot.apply_action(actions)
