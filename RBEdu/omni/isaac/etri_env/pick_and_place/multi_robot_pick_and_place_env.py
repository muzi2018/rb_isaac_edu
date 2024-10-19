from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.custom_controller import CustomPickPlaceController
from omni.isaac.etri_env import MultiEnvRandomizer

import numpy as np
import omni


class MultiRobotPickandPlaceEnv(MultiEnvRandomizer):

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

        # cur_robot = self._robots[index]
        # end_effector_initial_height = self._task_config["end_effector_initial_height"]
        
        # if "events_dt" in self._task_config:
        #     events_dt = self._task_config["events_dt"]
        # else:
        #     events_dt = None

        # try:
        #     controller = CustomPickPlaceController(
        #         name=f"pick_place_controller_{index}",
        #         gripper=cur_robot.gripper,
        #         robot_articulation=cur_robot,
        #         end_effector_initial_height=end_effector_initial_height,
        #         robot_type=self._robot_type,
        #         events_dt=events_dt,
        #     )
        # except Exception as e:
        #     print(f"Error: {e}")
        #     return

        # if self._task_type == "multi_pick_and_place":
        #     self._picking_object_names = self._task_config["picking_object_names"]
        #     self._picking_event = 0
        #     self._picking_events.append(self._picking_event)
        #     self._num_picking = len(self._picking_object_names)
        #     self._placing_positions = np.array(self._task_config["placing_positions"])
        # elif self._task_type == "pick_and_place":
        #     self._picking_object_name = self._task_config["picking_object_name"]
        #     self._placing_position = np.array(self._task_config["placing_position"])

        # self._end_effector_offset = np.array(self._task_config["end_effector_offset"])
        # self._object_offset = np.array(self._task_config["object_offset"])

        # self._picking_offset = np.array(self._task_config["picking_offset"])
        # self._picking_orientation = np.array(self._task_config["picking_orientation"])
        # self._placing_offset = np.array(self._task_config["placing_offset"])
        # self._placing_orientation = np.array(self._task_config["placing_orientation"])

        # self._controllers.append(controller)


    def add_multi_robot_task(self, index, env_offset):
        super().add_multi_robot_task(index, env_offset)
        self._multi_robot_task_config = self._task_config["multi_robot_task_config"]

        self._multi_task_len = len(self._multi_robot_task_config)
        
        for i, task_config in enumerate(self._multi_robot_task_config):
            robo_index = self._multi_task_len * index + i
            cur_robot = self._robots[robo_index]

            end_effector_initial_height = task_config["end_effector_initial_height"]

            if "events_dt" in task_config:
                events_dt = task_config["events_dt"]
            else:
                events_dt = None

            try:
                controller = CustomPickPlaceController(
                    name=f"pick_place_controller_{robo_index}",
                    gripper=cur_robot.gripper,
                    robot_articulation=cur_robot,
                    end_effector_initial_height=end_effector_initial_height,
                    # self._robot_type will be the latest robot type
                    robot_type=self._robot_type,
                    events_dt=events_dt,
                )
            except Exception as e:
                print(f"Error: {e}")
                return

            if self._verbose:
                print(f"[MultiRobotPickandPlaceEnv] Add multi robot controller Added! Env : {index} / Robot : {i}")
            
            self._controllers.append(controller)


    async def setup_post_load(self):
        await super().setup_post_load()
        return


    def forward(self):
        super().forward()

        for env_i, env_offset in enumerate(self._env_offsets):
            for task_i, task_config in enumerate(self._multi_robot_task_config):
                robo_index = self._multi_task_len * env_i + task_i
                cur_robot = self._robots[robo_index]

                if self._save_count < 10:
                    cur_robot.gripper.open()

                picking_object_name = task_config["picking_object_name"]
                placing_position = np.array(task_config["placing_position"])
                end_effector_offset = np.array(task_config["end_effector_offset"])
                object_offset = np.array(task_config["object_offset"])

                picking_offset = np.array(task_config["picking_offset"])
                picking_orientation = np.array(task_config["picking_orientation"])
                placing_offset = np.array(task_config["placing_offset"])
                placing_orientation = np.array(task_config["placing_orientation"])

                if "cloth" in picking_object_name.lower() or "deformable" in picking_object_name.lower():
                    obj_prim_path = f"/World/Env_{env_i}/{picking_object_name}/Meshes/object_mesh"

                    stage = omni.usd.get_context().get_stage()
                    obj_prim = stage.GetPrimAtPath(obj_prim_path)

                    obj_pose = np.array(omni.usd.get_world_transform_matrix(obj_prim).ExtractTranslation())
                    obj_pose_x, obj_pose_y, obj_pose_z = obj_pose[0], obj_pose[1], obj_pose[2]
                    object_position = np.array([obj_pose_x, obj_pose_y, obj_pose_z])
                else:
                    picking_object_name_geom = f"Env_{env_i}_{picking_object_name}"
                    object_geom = self._world.scene.get_object(picking_object_name_geom)
                    object_position, _ = object_geom.get_world_pose()

                picking_position = object_position + object_offset
                placing_position = env_offset + placing_position + object_offset
                # print(f"env_offset: {env_offset}")
                # print(f"self._placing_position: {self._placing_position}")
                # print(f"self._object_offset: {self._object_offset}")

                # robot frame to world frame 
                picking_orientation_world = rot_matrix_to_quat(
                    quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(picking_orientation)
                )
                placing_orientation_world = rot_matrix_to_quat(
                    quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(placing_orientation)
                )

                actions = self._controllers[robo_index].forward(
                    picking_position=picking_position,
                    placing_position=placing_position,
                    current_joint_positions=cur_robot.get_joint_positions(),
                    picking_offset=picking_offset,
                    placing_offset=placing_offset,
                    end_effector_offset=end_effector_offset,
                    picking_end_effector_orientation=picking_orientation_world,
                    placing_end_effector_orientation=placing_orientation_world,
                )
                if self._controllers[robo_index].is_done():
                    print(f"Env {env_i} Robot {task_i} Done!")
                    # self._controllers[robo_index].reset()
                else:
                    cur_robot.apply_action(actions)

        # #     if self._task_type == "multi_pick_and_place":
        # #         if self._picking_events[i] >= self._num_picking:
        # #             return
        # #         # print(f"self._picking_events[i]: {self._picking_events[i]}")
        # #         self._picking_object_name = self._picking_object_names[self._picking_events[i]]
        # #         self._placing_position = self._placing_positions[self._picking_events[i]]

        #     if "cloth" in self._picking_object_name.lower() or "deformable" in self._picking_object_name.lower():
        #         # print("deformable object")
        #         obj_prim_path = f"/World/Env_{i}/{self._picking_object_name}/Meshes/object_mesh"
                
        #         stage = omni.usd.get_context().get_stage()
        #         obj_prim = stage.GetPrimAtPath(obj_prim_path)

        #         obj_pose = np.array(omni.usd.get_world_transform_matrix(obj_prim).ExtractTranslation())
        #         obj_pose_x, obj_pose_y, obj_pose_z = obj_pose[0], obj_pose[1], obj_pose[2]
        #         object_position = np.array([obj_pose_x, obj_pose_y, obj_pose_z])
        #         # print(f"object_position: {object_position}")
        #     else:
        #         picking_object_name = f"Env_{i}_{self._picking_object_name}"
        #         object_geom = self._world.scene.get_object(picking_object_name)
        #         object_position, _ = object_geom.get_world_pose()

        #     picking_position = object_position + self._object_offset
            
        #     env_offset = self._env_offsets[i]
        #     placing_position = env_offset + self._placing_position + self._object_offset
        #     # print(f"env_offset: {env_offset}")
        #     # print(f"self._placing_position: {self._placing_position}")
        #     # print(f"self._object_offset: {self._object_offset}")

        #     # robot frame to world frame 
        #     picking_orientation_world = rot_matrix_to_quat(
        #         quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._picking_orientation)
        #     )
        #     placing_orientation_world = rot_matrix_to_quat(
        #         quat_to_rot_matrix(self._robot_rotation_quat) @ quat_to_rot_matrix(self._placing_orientation)
        #     )

        #     actions = self._controllers[i].forward(
        #         picking_position=picking_position,
        #         placing_position=placing_position,
        #         current_joint_positions=robot.get_joint_positions(),
        #         picking_offset=self._picking_offset,
        #         placing_offset=self._placing_offset,
        #         end_effector_offset=self._end_effector_offset,
        #         picking_end_effector_orientation=picking_orientation_world,
        #         placing_end_effector_orientation=placing_orientation_world,
        #     )
        #     if self._controllers[i].is_done():
        #         self._controllers[i].reset()
        #     else:
        #         robot.apply_action(actions)
