from typing import Optional

import carb
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root


class UR102F(Robot):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "ur10_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        attach_gripper (bool, optional): [description]. Defaults to False.

    Raises:
        NotImplementedError: [description]
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur10_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
                self._server_root = get_url_root(default_asset_root)
                self._root_path = get_assets_root_path()
                self.UR_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR10/ur10_2f85.usd"
                
                add_reference_to_stage(usd_path=self.UR_PATH, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if attach_gripper:
            gripper_dof_names = [
                "finger_joint", 
                "right_outer_knuckle_joint",
            ]
            gripper_open_position = np.array([0.0, 0.0])
            gripper_closed_position = np.array([40.0, 40.0])
            deltas = np.array([-40.0, -40.0])
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )
        self._attach_gripper = attach_gripper
        return

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

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
            SurfaceGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self.disable_gravity()
        self._end_effector.initialize(physics_sim_view)
        # print(f"self.dof_names : {self.dof_names}")
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
        Robot.post_reset(self)
        self._gripper.post_reset()
        return
