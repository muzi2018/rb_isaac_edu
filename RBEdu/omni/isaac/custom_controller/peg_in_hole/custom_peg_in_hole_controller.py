import typing
from typing import List, Optional

import numpy as np
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.universal_robots.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.stage import get_stage_units

from omni.isaac.ur5 import RMPFlowController as UR5RMPFlowController
from omni.isaac.zeus import RMPFlowController as ZeusRMPFlowController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController as FrankaRMPFlowController
from omni.isaac.universal_robots.controllers.rmpflow_controller import RMPFlowController as UR102F85RMPFlowController


class CustomPegInHoleController(manipulators_controllers.PickPlaceController):
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
        events_dt: Optional[List[float]] = None,
        robot_type: str = "franka",
    ) -> None:
        BaseController.__init__(self, name=name)

        self._t = 0
        self._h0 = None
        self._robot = robot_articulation
        self._robot_type = robot_type

        if events_dt is not None:
            if not isinstance(events_dt, np.ndarray) and not isinstance(events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(events_dt, np.ndarray):
                events_dt = events_dt.tolist()
            if len(events_dt) > 8:
                raise Exception("events dt length must be less than 8")
            self._events_dt_pick = events_dt[:4]
            self._events_dt_peg = events_dt[4:]
        else:
            # Fast mode
            self._events_dt_pick = [
                0.01,
                0.01,
                0.1,
                0.1,
            ]
            self._events_dt_peg = [
                0.02,
                0.005,
                0.005,
            ]

        if self._robot_type == "franka":
            cspace_controller = FrankaRMPFlowController(
                name="pickplace_cspace_controller", 
                robot_articulation=self._robot
            )
        elif self._robot_type == "zeus":
            cspace_controller = ZeusRMPFlowController(
                name="pickplace_cspace_controller", 
                robot_articulation=self._robot
            )
        elif self._robot_type == "ur5":
            cspace_controller = UR5RMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )
        elif self._robot_type == "ur10":
            cspace_controller = UR102F85RMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._is_done = False
        self._pause = False

        self._state = 0
        self._event = 0
        self._peg_iterations = 0
        self._num_peg_points = 0
        return


    def forward(
        self,
        picking_position: np.ndarray,
        end_effector_initial_height: float = 1.3,
        picking_offset: np.ndarray = None,
        pegging_offset: np.ndarray = None,
        pegging_positions: np.ndarray = None,
        current_joint_positions: np.ndarray = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        
        print(f"[PegInHoleController] self._state : {self._state} / self._event: {self._event}")

        if self._state == 0:
            self._num_peg_points = len(pegging_positions)
            if end_effector_offset is None:
                end_effector_offset = np.array([0, 0, 0])
            if self._pause or self.is_done():
                self.pause()
                target_joint_positions = [None] * current_joint_positions.shape[0]
                return ArticulationAction(joint_positions=target_joint_positions)
            if self._event == 2:
                target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            elif self._event == 3:
                target_joint_positions = self._gripper.forward(action="close")
            else:
                if self._event in [0, 1]:
                    self._current_target_x = picking_position[0]
                    self._current_target_y = picking_position[1]
                    self._h0 = picking_position[2]
                interpolated_xy = self._get_interpolated_xy(
                    pegging_positions[0][0], pegging_positions[0][1], self._current_target_x, self._current_target_y
                )
                target_height = self._get_target_hs(
                    pegging_positions[0][2],
                    height_offset=picking_offset[2]
                )
                if self._event == 0:
                    position_target = np.array(
                        [
                            self._current_target_x,
                            self._current_target_y,
                            target_height + end_effector_offset[2],
                        ]
                    )
                else:
                    position_target = np.array(
                        [
                            interpolated_xy[0] + end_effector_offset[0],
                            interpolated_xy[1] + end_effector_offset[1],
                            target_height + end_effector_offset[2],
                        ]
                    )
                if end_effector_orientation is None:
                    end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=position_target, 
                    target_end_effector_orientation=end_effector_orientation
                )
            self._t += self._events_dt_pick[self._event]
        elif self._state == 1:
            print(f"self._peg_iterations: {self._peg_iterations}")
            if self._peg_iterations == 0:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = end_effector_initial_height
            else:
                self._current_target_x = pegging_positions[self._peg_iterations - 1][0]
                self._current_target_y = pegging_positions[self._peg_iterations - 1][1]
                self._h0 = pegging_positions[self._peg_iterations - 1][2]
            interpolated_xy = self._get_interpolated_xy(
                pegging_positions[self._peg_iterations][0], pegging_positions[self._peg_iterations][1], 
                self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(
                pegging_positions[self._peg_iterations][2],
                height_offset=pegging_offset[2]
            )
            if self._event == 0:
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0],
                        interpolated_xy[1] + end_effector_offset[1],
                        target_height + end_effector_offset[2],
                    ]
                )
            else:
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0],
                        interpolated_xy[1] + end_effector_offset[1],
                        target_height + end_effector_offset[2],
                    ]
                )

            print(f"target_height: {target_height}")

            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
            self._t += self._events_dt_peg[self._event]
        elif self._state == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        if self._t >= 1.0:
            self._event += 1
            self._t = 0
            if self._state == 0 and self._event > 3:
                self._state = 1
                self._event = 0
            if self._state == 1 and self._event > 2:
                self._event = 0
                self._peg_iterations += 1
                if self._peg_iterations >= self._num_peg_points:
                    print("Peg in Hole done!")
                    self._is_done = True
                    self._state = 2
                    self._event = 0
            # print(f"self._num_peg_points: {self._num_peg_points}")
            # print(f"self._state: {self._state} / self._event: {self._event} / self._peg_iterations: {self._peg_iterations}")
        
        return target_joint_positions

    def _get_alpha(self):
        if self._state == 0 and self._event < 4:
            return 0
        elif self._state == 1 and self._event == 0:
            return 0
        elif self._state == 1 and self._event == 1:
            return self._mix_sin(self._t)
        elif self._state == 1 and self._event == 2:
            return 1.0
        else:
            raise ValueError()

    def _get_target_hs(self, target_height, height_offset=0.0):
        if self._state == 0 and self._event == 0:
            h = self._h0 + height_offset
        elif self._state == 0 and self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(
                self._h0 + height_offset, self._h0, a
            )
        elif self._state == 0 and self._event == 3:
            h = self._h0
        elif self._state == 1 and self._event == 0:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(
                self._h0, self._h0 + height_offset, a
            )
        elif self._state == 1 and self._event == 1:
            print(f"self._h0: {self._h0} / height_offset: {height_offset}")
            h = self._h0 + height_offset
        elif self._state == 1 and self._event == 2:
            h = self._combine_convex(
                self._h0 + height_offset, target_height, self._mix_sin(self._t)
            )
        else:
            raise ValueError()
        return h


    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        return self._is_done
    

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._state = 0
        self._event = 0
        self._peg_iterations = 0
        self._num_peg_points = 0
        self._t = 0
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        return