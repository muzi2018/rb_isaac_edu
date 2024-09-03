import typing

import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

from omni.isaac.ur5 import RMPFlowController as UR5RMPFlowController
from omni.isaac.zeus import RMPFlowController as ZeusRMPFlowController

from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController as FrankaRMPFlowController
from omni.isaac.universal_robots.controllers.rmpflow_controller import RMPFlowController as UR102F85RMPFlowController


class CustomPickPlaceController(BaseController):
    """
    A simple pick and place state machine for tutorials

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        gripper: Gripper,
        robot_articulation: Articulation,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        robot_type: str = "franka",
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h0 = None
        self._robot = robot_articulation
        self._robot_type = robot_type
        self._position_target = None
        self._events_dt = events_dt

        # if self._events_dt is not None:
            # self._events_dt = [0.003, 0.007, 0.2, 0.2, 0.01, 0.005, 0.002, 0.05, 0.008, 0.01]
        # else:
        if self._events_dt is not None:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")

        if self._robot_type == "franka":
            cspace_controller = FrankaRMPFlowController(
                name="pickplace_cspace_controller", 
                robot_articulation=self._robot
            )
            if self._events_dt is None:
                # During battery pack pp
                self._events_dt = [0.003, 0.007, 0.2, 0.2, 0.01, 0.002, 0.002, 0.05, 0.008, 0.01]
        elif self._robot_type == "zeus":
            cspace_controller = ZeusRMPFlowController(
                name="pickplace_cspace_controller", 
                robot_articulation=self._robot
            )
            if self._events_dt is None:
                # big robot requires more time to settle
                self._events_dt = [0.002, 0.007, 0.2, 0.2, 0.01, 0.005, 0.002, 0.05, 0.008, 0.01]
        elif self._robot_type == "ur5":
            cspace_controller = UR5RMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )
            if self._events_dt is None:
                self._events_dt = [0.003, 0.007, 0.2, 0.2, 0.01, 0.005, 0.002, 0.05, 0.008, 0.01]
        elif self._robot_type == "ur10":
            cspace_controller = UR102F85RMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )
            if self._events_dt is None:
                # big robot requires more time to settle
                self._events_dt = [0.003, 0.007, 0.2, 0.2, 0.01, 0.005, 0.002, 0.05, 0.008, 0.01]

        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def get_position_target(self):
        return self._position_target

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        picking_offset: typing.Optional[np.ndarray] = None,
        placing_offset: typing.Optional[np.ndarray] = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        picking_end_effector_orientation: typing.Optional[np.ndarray] = None,
        placing_end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        
        print(f"[PickPlaceController] event: {self._event}")
        
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
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            if self._event == 0:
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0] + picking_offset[0],
                        interpolated_xy[1] + end_effector_offset[1] + picking_offset[1],
                        # TODO: check for different robots
                        self._h0 + picking_offset[2],
                    ]
                )
            elif self._event == 8:
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0] + placing_offset[0],
                        interpolated_xy[1] + end_effector_offset[1] + placing_offset[1],
                        placing_position[2] + placing_offset[2],
                    ]
                )
            elif self._event == 9:
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0] + placing_offset[0],
                        interpolated_xy[1] + end_effector_offset[1] + placing_offset[1],
                        self._h1 + placing_offset[2],
                    ]
                )
            else:
                position_target = np.array(
                    [
                        interpolated_xy[0] + end_effector_offset[0],
                        interpolated_xy[1] + end_effector_offset[1],
                        # TODO Check this
                        target_height, # + end_effector_offset[2],
                    ]
                )

            if self._event <= 5 or self._event == 9:
                if picking_end_effector_orientation is None:
                    target_picking_end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                target_picking_end_effector_orientation = picking_end_effector_orientation
            else:
                if placing_end_effector_orientation is None:
                    target_picking_end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                target_picking_end_effector_orientation = placing_end_effector_orientation
            
            # print(f"position_target: {position_target} / target_picking_end_effector_orientation: {target_picking_end_effector_orientation}")
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, 
                target_end_effector_orientation=target_picking_end_effector_orientation
            )
            self._position_target = position_target
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            # TODO : check other robots for this
            if self._robot_type == "zeus":
                h = self._h1
            else:
                h = self._h0
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._h0
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            # h = self._h1
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h0
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
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

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return
