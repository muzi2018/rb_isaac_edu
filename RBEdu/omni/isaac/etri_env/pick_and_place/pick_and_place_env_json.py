from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.custom_controller import CustomPickPlaceController
from omni.isaac.etri_env import MultiEnvRandomizer

import numpy as np
import omni


class MultiPickandPlaceEnv(MultiEnvRandomizer):

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

        if self._task_type.lower() == "nothing":
            print("Task Type is Nothing")
            return

        cur_robot = self._robots[index]
        end_effector_initial_height = self._task_config["end_effector_initial_height"]
        
        if "events_dt" in self._task_config:
            events_dt = self._task_config["events_dt"]
        else:
            events_dt = None

        try:
            controller = CustomPickPlaceController(
                name=f"pick_place_controller_{index}",
                gripper=cur_robot.gripper,
                robot_articulation=cur_robot,
                end_effector_initial_height=end_effector_initial_height,
                robot_type=self._robot_type,
                events_dt=events_dt,
            )
        except Exception as e:
            print(f"Error: {e}")
            return

        if self._task_type == "multi_pick_and_place":
            self._picking_object_names = self._task_config["picking_object_names"]
            self._picking_event = 0
            self._picking_events.append(self._picking_event)
            self._num_picking = len(self._picking_object_names)
            self._placing_positions = np.array(self._task_config["placing_positions"])
        elif self._task_type == "pick_and_place":
            self._picking_object_name = self._task_config["picking_object_name"]
            self._placing_position = np.array(self._task_config["placing_position"])

        self._end_effector_offset = np.array(self._task_config["end_effector_offset"])
        self._object_offset = np.array(self._task_config["object_offset"])

        self._picking_offset = np.array(self._task_config["picking_offset"])
        self._picking_orientation = np.array(self._task_config["picking_orientation"])
        self._placing_offset = np.array(self._task_config["placing_offset"])
        self._placing_orientation = np.array(self._task_config["placing_orientation"])

        self._controllers.append(controller)


    async def setup_post_load(self):
        await super().setup_post_load()
        return


    def forward(self):
        super().forward()

        for i, robot in enumerate(self._robots):

            if self._task_type.lower() == "nothing":
                print("Task Type is Nothing")
                return

            # exception : franka normal에서 벌어지지 않는 경우 발생
            if self._save_count < 10:
                robot.gripper.open()

            if self._task_type == "multi_pick_and_place":
                if self._picking_events[i] >= self._num_picking:
                    return
                # print(f"self._picking_events[i]: {self._picking_events[i]}")
                self._picking_object_name = self._picking_object_names[self._picking_events[i]]
                self._placing_position = self._placing_positions[self._picking_events[i]]

            if "cloth" in self._picking_object_name.lower() or "deformable" in self._picking_object_name.lower():
                # print("deformable object")
                obj_prim_path = f"/World/Env_{i}/{self._picking_object_name}/Meshes/object_mesh"
                
                stage = omni.usd.get_context().get_stage()
                obj_prim = stage.GetPrimAtPath(obj_prim_path)

                obj_pose = np.array(omni.usd.get_world_transform_matrix(obj_prim).ExtractTranslation())
                obj_pose_x, obj_pose_y, obj_pose_z = obj_pose[0], obj_pose[1], obj_pose[2]
                object_position = np.array([obj_pose_x, obj_pose_y, obj_pose_z])
                # print(f"object_position: {object_position}")
            elif "pre_dynamics" in self._picking_object_name.lower():
                obj_prim_path = f"/World/Env_{i}/{self._picking_object_name}"
                
                stage = omni.usd.get_context().get_stage()
                obj_prim = stage.GetPrimAtPath(obj_prim_path)

                obj_pose = np.array(omni.usd.get_world_transform_matrix(obj_prim).ExtractTranslation())
                obj_pose_x, obj_pose_y, obj_pose_z = obj_pose[0], obj_pose[1], obj_pose[2]
                object_position = np.array([obj_pose_x, obj_pose_y, obj_pose_z])
            else:
                picking_object_name = f"Env_{i}_{self._picking_object_name}"
                object_geom = self._world.scene.get_object(picking_object_name)
                object_position, _ = object_geom.get_world_pose()

            picking_position = object_position + self._object_offset
            
            env_offset = self._env_offsets[i]
            placing_position = env_offset + self._placing_position + self._object_offset
            # print(f"env_offset: {env_offset}")
            # print(f"self._placing_position: {self._placing_position}")
            # print(f"self._object_offset: {self._object_offset}")

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
                if self._task_type == "multi_pick_and_place":
                    self._picking_events[i] += 1
                    # print(f"self._picking_events[i]: {self._picking_events[i]}")
                    print(f"self._num_picking: {self._num_picking}")
                    if self._picking_events[i] >= self._num_picking:
                        print(f"[MultiPickandPlaceEnv] done picking and placing")
                    else:
                        self._controllers[i].reset()
            else:
                robot.apply_action(actions)
