import typing
from typing import List, Optional

import carb
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units

from omni.isaac.robotiq_3f_gripper import Robotiq3F
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper


class Franka3F(Robot):

    def __init__(
        self,
        prim_path: str,
        name: str = "franka_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
        gripper_mode: str = "basic",
        custom_open_positions: Optional[np.ndarray] = None,
        custom_closed_positions: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        self._gripper_mode = gripper_mode

        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
                self._server_root = get_url_root(default_asset_root)        
                self._root_path = get_assets_root_path()
                # self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/Franka/franka_3f.usd"
                self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/Franka/franka_3f_pp.usd"

                add_reference_to_stage(usd_path=self.ROBOT_PATH, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_link8"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_link8"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if attach_gripper:
            
            gripper_dof_names = [
                "palm_finger_1_joint",
                "palm_finger_2_joint",
                "finger_2_joint_1",
                "finger_2_joint_2",
                "finger_2_joint_3",
                "finger_1_joint_1", 
                "finger_1_joint_2",
                "finger_1_joint_3",
                "finger_middle_joint_1",
                "finger_middle_joint_2",
                "finger_middle_joint_3",
            ]

            deltas = None
            gripper_open_position = np.array([
                -10.0, 10.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])
            gripper_closed_position = np.array([
                -10.0, 10.0,
                70.0, 90.0, 0.0,
                70.0, 90.0, 0.0,
                70.0, 90.0, 0.0,
            ])

            if self._gripper_mode == "basic":
                deltas = np.array([
                    0.0, 0.0,
                    -70.0, -90.0, 0.0,
                    -70.0, -90.0, 0.0,
                    -70.0, -90.0, 0.0,
                ])
            elif self._gripper_mode == "wide":
                deltas = None
                gripper_open_position = np.array([
                    10.0, -10.0,
                    -50.0, 0.0, 50.0,
                    -50.0, 0.0, 50.0,
                    -50.0, 0.0, 50.0,
                ])
                gripper_closed_position = np.array([
                    10.0, -10.0,
                    50.0, 0.0, -50.0,
                    50.0, 0.0, -50.0,
                    50.0, 0.0, -50.0,
                ])
            elif self._gripper_mode == "pinch":
                deltas = None
                gripper_open_position = np.array([
                    -10.0, 10.0,
                    -50.0, 0.0, 50.0,
                    -50.0, 0.0, 50.0,
                    -50.0, 0.0, 50.0,
                ])
                gripper_closed_position = np.array([
                    -10.0, 10.0,
                    50.0, 0.0, -50.0,
                    50.0, 0.0, -50.0,
                    50.0, 0.0, -50.0,
                ])

            elif self._gripper_mode == "screw":
                deltas = np.array([
                    0.0, 0.0,
                    -50.0, 0.0, 50.0,
                    -50.0, 0.0, 50.0,
                    -50.0, 0.0, 50.0,
                ])

                gripper_open_position = np.array([
                    -10.0, 10.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                ])
                gripper_closed_position = np.array([
                    -10.0, 10.0,
                    50.0, 0.0, -50.0,
                    50.0, 0.0, -50.0,
                    50.0, 0.0, -50.0,
                ])

            elif self._gripper_mode == "custom":
                deltas = None
                gripper_open_position = np.array(custom_open_positions)
                gripper_closed_position = np.array(custom_closed_positions)

            self._gripper = Robotiq3F(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
                gripper_mode=self._gripper_mode,
            )
        self._attach_gripper = attach_gripper
        return

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        print(f"self.dof_names : {self.dof_names}")

        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return
