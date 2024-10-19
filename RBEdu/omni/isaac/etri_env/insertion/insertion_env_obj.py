from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.custom_controller import CustomInsertionController

from ..multi_env_randomizer import MultiEnvRandomizer

import numpy as np 
import logging
import carb
import omni


class MultiInsertionObjEnv(MultiEnvRandomizer):
    
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

        if self._task_type == "insertion":
            end_effector_initial_height = self._task_config["end_effector_initial_height"]

            controller = CustomInsertionController(
                name=f"insertion_controller_{index}",
                gripper=cur_robot.gripper,
                robot_articulation=cur_robot,
                end_effector_initial_height=end_effector_initial_height,
                robot_type=self._robot_type,
                events_dt=events_dt,
            )

            self._end_effector_offset = np.array(self._task_config["end_effector_offset"])

            self._picking_object_name = self._task_config["picking_object_name"][0]
            self._insert_object_name = self._task_config["insert_object_name"][0]

            self._picking_object_offset = np.array(self._task_config["picking_object_offset"])
            self._insert_object_offset = np.array(self._task_config["insert_object_offset"])

            self._picking_offset = np.array(self._task_config["picking_offset"])
            self._picking_orientation = np.array(self._task_config["picking_orientation"])
            
            self._insert_offset = np.array(self._task_config["insert_offset"])
            self._insert_orientation = np.array(self._task_config["insert_orientation"])
            # self._insert_position = np.array(self._task_config["insert_position"])

        self._controllers.append(controller)


    async def setup_post_load(self):
        await super().setup_post_load()
        return
    
    def forward(self):
        super().forward()

        for i, robot in enumerate(self._robots):

            if self._save_count < 10:
                robot.gripper.open()

            # Empty Insertion Here
            if self._controllers[i]._event == 8:
                object_mesh_paths = []
                if "cable" in self._picking_object_name.lower():
                    prim_path = f"/World/Env_{i}/{self._picking_object_name}"
                    fc_prim_path = prim_path + "/Front_Connector"
                    bc_prim_path = prim_path + "/Back_Connector"

                    object_mesh_paths.append(fc_prim_path)
                    object_mesh_paths.append(bc_prim_path)
                else:
                    insert_prim_path = f"/World/Env_{i}/{self._picking_object_name}"
                    object_mesh_paths.append(insert_prim_path)
                
                print(f"object_mesh_paths: {object_mesh_paths}")
                for object_mesh_path in object_mesh_paths:
                    UsdPhysics.RigidBodyAPI(
                        self._stage.GetPrimAtPath(object_mesh_path)
                    ).CreateKinematicEnabledAttr(True)

            if "cable" in self._picking_object_name.lower():
                fc_prim_path = f"/World/Env_{i}/{self._picking_object_name}/Front_Connector"
                fc_prim = self._stage.GetPrimAtPath(fc_prim_path)
                fc_pose = np.array(omni.usd.get_world_transform_matrix(fc_prim).ExtractTranslation())
                fc_pose_x, fc_pose_y, fc_pose_z = fc_pose[0], fc_pose[1], fc_pose[2]
                picking_object_position = np.array([fc_pose_x, fc_pose_y, fc_pose_z])
            else:
                picking_object_name = f"Env_{i}_{self._picking_object_name}"
                picking_object_geom = self._world.scene.get_object(picking_object_name)
                picking_object_position, _ = picking_object_geom.get_world_pose()

            if "port" in self._insert_object_name.lower():
                insert_object_path = f"/World/Env_{i}/{self._insert_object_name}"
                insert_object_prim = self._stage.GetPrimAtPath(insert_object_path)
                io_pose = np.array(omni.usd.get_world_transform_matrix(insert_object_prim).ExtractTranslation())
                io_pose_x, io_pose_y, io_pose_z = io_pose[0], io_pose[1], io_pose[2]
                insert_object_position = np.array([io_pose_x, io_pose_y, io_pose_z])
            else:
                insert_object_name = f"Env_{i}_{self._insert_object_name}"
                insert_object_geom = self._world.scene.get_object(insert_object_path)
                insert_object_position, _ = insert_object_geom.get_world_pose()

            # print(f"picking_object_position: {picking_object_position} / insert_object_position: {insert_object_position}")

            picking_position = picking_object_position + self._picking_object_offset
            placing_position = insert_object_position + self._insert_object_offset

            # robot frame to world frame 
            picking_orientation_world = rot_matrix_to_quat(
                quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._picking_orientation)
            )
            placing_orientation_world = rot_matrix_to_quat(
                quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._insert_orientation)
            )

            actions = self._controllers[i].forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=robot.get_joint_positions(),
                picking_offset=self._picking_offset,
                placing_offset=self._insert_offset,
                end_effector_offset=self._end_effector_offset,
                picking_end_effector_orientation=picking_orientation_world,
                placing_end_effector_orientation=placing_orientation_world,
            )
            if self._controllers[i].is_done():
                print(f"[MultiInsertionEnvJSON {i}] done picking and placing")
            else:
                robot.apply_action(actions)
