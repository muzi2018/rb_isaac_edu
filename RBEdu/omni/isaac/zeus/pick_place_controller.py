from typing import List, Optional

import numpy as np
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper

from .controllers.rmpflow import RMPFlowController

class PickPlaceController(manipulators_controllers.PickPlaceController):
    """[summary]

    Args:
        name (str): [description]
        surface_gripper (SurfaceGripper): [description]
        robot_articulation(Articulation): [description]
        events_dt (Optional[List[float]], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        end_effector_initial_height: float = None,
        events_dt: Optional[List[float]] = None,
        fast_movement=False
    ) -> None:
        if events_dt is None:
            if fast_movement is False:
                events_dt = [
                    0.01,
                    0.0035,
                    0.01,
                    1.0,
                    0.008,
                    0.005,
                    0.005,
                    1,
                    0.01,
                    0.08
                ]
            else:
                # Fast mode
                events_dt = [
                    0.01,
                    0.01,
                    0.1,
                    1.0,
                    0.02,
                    0.02,
                    0.005,
                    1,
                    0.02,
                    0.08
                ]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", 
                robot_articulation=robot_articulation, 
                # attach_gripper=True,
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
        return

    # Useless Code
    # 
    # def forward(
    #     self,
    #     picking_position: np.ndarray,
    #     placing_position: np.ndarray,
    #     current_joint_positions: np.ndarray,
    #     end_effector_offset: Optional[np.ndarray] = None,
    #     end_effector_orientation: Optional[np.ndarray] = None,
    # ) -> ArticulationAction:
    #     """[summary]

    #     Args:
    #         picking_position (np.ndarray): [description]
    #         placing_position (np.ndarray): [description]
    #         current_joint_positions (np.ndarray): [description]
    #         end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    #         end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.

    #     Returns:
    #         ArticulationAction: [description]
    #     """
    #     if end_effector_orientation is None:
    #         end_effector_orientation = euler_angles_to_quat(np.array([0, 0, 0]))
    #     return super().forward(
    #         picking_position,
    #         placing_position,
    #         current_joint_positions,
    #         end_effector_offset=end_effector_offset,
    #         end_effector_orientation=end_effector_orientation,
    #     )
