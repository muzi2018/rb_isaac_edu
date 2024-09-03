from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.custom_controller import CustomPegInHoleController

from omni.isaac.etri_env import MultiEnvRandomizer

import numpy as np
import logging
import omni


class PegInHoleEnv(MultiEnvRandomizer):

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
        super().add_task(index, env_offset)

        try:
            cur_robot = self._robots[index]
        except Exception as e:
            logging.error(f"[InsertionEnv] Failed to get robot at index {index}")
            return
        
        if "events_dt" in self._task_config:
            events_dt = self._task_config["events_dt"]
        else:
            events_dt = None

        try:
            controller = CustomPegInHoleController(
                name=f"peg_in_hole_controller_{index}",
                gripper=cur_robot.gripper,
                robot_articulation=cur_robot,
                events_dt=events_dt,
                robot_type=self._robot_type,
            )
        except Exception as e:
            print(f"Error: {e}")
            return

        self._picking_object_name = self._task_config["picking_object_name"]
        # print(f"self._picking_object_name:  {self._picking_object_name}")
        self._object_offset = np.array(self._task_config["object_offset"])
        self._pegging_positions = np.array(self._task_config["pegging_positions"])

        self._picking_offset = np.array(self._task_config["picking_offset"])
        self._pegging_offset = np.array(self._task_config["pegging_offset"])
        
        self._end_effector_initial_height = self._task_config["end_effector_initial_height"]
        self._end_effector_offset = np.array(self._task_config["end_effector_offset"])
        self._end_effector_orientation = np.array(self._task_config["end_effector_orientation"])
        # self._picking_orientation = np.array(self._task_config["picking_orientation"])
        # self._placing_orientation = np.array(self._task_config["placing_orientation"])
        
        self._controllers.append(controller)


    def setup_robots(self, index):
        super().setup_robots(index)

        robot = self._robots[index]

        if self._robot_type == "franka" and self._gripper_type == "normal":
            kps = np.array([6000000.0, 600000.0, 6000000.0, 600000.0, 25000.0, 15000.0, 25000.0, 
                            15000.0, 15000.0])
            kds = np.array([600000.0, 60000.0, 300000.0, 30000.0, 3000.0, 3000.0, 3000.0, 
                            6000.0, 6000.0])
            robot.get_articulation_controller().set_gains(kps=kps, kds=kds, save_to_usd=True)

        elif self._robot_type == "franka" and self._gripper_type == "2f85":
            kps = np.array([6000000.0, 600000.0, 6000000.0, 600000.0, 25000.0, 15000.0, 25000.0, 
                            15000.0, 15000.0, 0.0, 0.0, 0.0, 0.0])
            kds = np.array([600000.0, 60000.0, 300000.0, 30000.0, 3000.0, 3000.0, 3000.0, 
                            1500.0, 1500.0, 0.0, 0.0, 0.0, 0.0])
            robot.get_articulation_controller().set_gains(kps=kps, kds=kds, save_to_usd=True)


        return


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
            pegging_positions = self._pegging_positions + env_offset + self._object_offset
            # print(f"env_offset: {env_offset}")
            # print(f"self._placing_position: {self._placing_position}")
            # print(f"self._object_offset: {self._object_offset}")

            # # robot frame to world frame 
            # picking_orientation_world = rot_matrix_to_quat(
            #     quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._picking_orientation)
            # )
            # placing_orientation_world = rot_matrix_to_quat(
            #     quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._placing_orientation)
            # )

            end_effector_orientation = rot_matrix_to_quat(
                quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._end_effector_orientation)
            )

            actions = self._controllers[i].forward(
                end_effector_initial_height=self._end_effector_initial_height,
                picking_position=picking_position,
                picking_offset=self._picking_offset,
                pegging_offset=self._pegging_offset,
                pegging_positions=pegging_positions,
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=self._end_effector_offset,
                end_effector_orientation=end_effector_orientation,
                # picking_end_effector_orientation=picking_orientation_world,
                # placing_end_effector_orientation=placing_orientation_world,
            )
            if self._controllers[i].is_done():
                print(f"[PegInHoleEnv {i}] done peg in hole")
            else:
                robot.apply_action(actions)
