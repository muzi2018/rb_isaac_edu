from omni.isaac.manipulators.controllers.pick_place_controller import PickPlaceController
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.robots.robot import Robot

from omni.isaac.ur5 import RMPFlowController as UR5RMPFlowController
from omni.isaac.zeus import RMPFlowController as ZeusRMPFlowController

from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController as FrankaRMPFlowController
from omni.isaac.universal_robots.controllers.rmpflow_controller import RMPFlowController as UR102F85RMPFlowController

import numpy as np
import typing

from .screw_controller import ScrewController

class NutBoltController(BaseController):
    """
    A state machine to tie nuts onto bolts with a vibrating table feeding the nuts

    - State 0: Pick and Place from pickup location on vibration table to different bolts
    - State 1: Screw nut onto bolt

    Args:
        name (str): Name id of the controller
        robot (Robot): Robot

    """

    def __init__(
            self, 
            name: str, 
            robot: Robot, 
            robot_type: str = "franka",
        ) -> None:
        BaseController.__init__(self, name=name)
        # Here
        self._event = 0
        self._robot = robot
        self._gripper = self._robot.gripper
        self._robot_type = robot_type
        
        if self._robot_type == "franka" or self._robot_type == "franka_normal":
            self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.4 / get_stage_units())
            self._cspace_controller = FrankaRMPFlowController(
                name="pickplace_cspace_controller", 
                robot_articulation=self._robot
            )
        elif self._robot_type == "ur5":
            # updated for ur6 0.4 > 0.45 > 0.5
            self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.5 / get_stage_units())
            self._cspace_controller = UR5RMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )
        elif self._robot_type == "ur10":
            # 0.4 > 0.5
            self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.5 / get_stage_units())
            self._cspace_controller = UR102F85RMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )
        elif self._robot_type == "zeus":
            self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.47 / get_stage_units())
            self._cspace_controller = ZeusRMPFlowController(
                name="pickplace_cspace_controller",
                robot_articulation=self._robot
            )
        print(f"self._end_effector_initial_height: {self._end_effector_initial_height}")


        self._pause = False

        # TODO pick_place_events_dt as parameter
        # pick_place_events_dt = [0.01, 0.008, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
        
        # Franka
        if self._robot_type == "franka" or self._robot_type == "franka_normal":
            pick_place_events_dt = [0.005, 0.005, 0.1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
        elif self._robot_type == "ur5":
            pick_place_events_dt = [0.01, 0.008, 0.1, 0.1, 0.05, 0.03, 0.01, 1, 0.008, 0.08]
        elif self._robot_type == "ur10":
            pick_place_events_dt = [0.01, 0.008, 0.1, 0.1, 0.05, 0.03, 0.01, 1, 0.008, 0.08]
        elif self._robot_type == "zeus":
            pick_place_events_dt = [0.008, 0.008, 0.1, 0.1, 0.05, 0.005, 0.005, 1, 0.008, 0.08]

        self._pick_place_controller = PickPlaceController(
            name="pickplace_controller",
            cspace_controller=self._cspace_controller,
            gripper=self._gripper,
            end_effector_initial_height=self._end_effector_initial_height,
            events_dt=pick_place_events_dt,
        )
        
        self._screw_controller = ScrewController(
            name=f"screw_controller", 
            cspace_controller=self._cspace_controller, 
            gripper=self._gripper,
            robot_type=self._robot_type,
        )

        self._i = 0
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

    def forward(
        self,
        initial_picking_position: np.ndarray,
        bolt_top: np.ndarray,
        gripper_to_nut_offset: np.ndarray,
        initial_end_effector_orientation: np.ndarray = None,
        x_offset: np.ndarray = 0.0,
        y_offset: np.ndarray = 0.0,
    ):
        """Runs the controller one step.

        Args:
            initial_picking_position (np.ndarray): initial nut position at table feeder
            bolt_top (np.ndarray):  bolt target position

        #"""
        if self.is_paused():
            return
        if self._event == 0:
            initial_effector_orientation = quat_to_euler_angles(self._gripper.get_world_pose()[1])
            initial_effector_orientation[2] = np.pi / 2
            
            # only franka
            # initial_end_effector_orientation = euler_angles_to_quat(initial_effector_orientation)
            
            if initial_end_effector_orientation is None:
                if self._robot_type == "franka" or self._robot_type == "franka_normal":
                    initial_end_effector_orientation = euler_angles_to_quat(np.array([
                        np.pi, 0.0, np.pi/2
                        # 0, np.pi, np.pi/2
                    ]))
                    # initial_end_effector_orientation: [4.32978028e-17 7.07106781e-01 7.07106781e-01 4.32978028e-17]
                elif self._robot_type == "ur5":
                    initial_end_effector_orientation = euler_angles_to_quat(np.array([
                        np.pi, 0.0, np.pi/2
                    ]))
                elif self._robot_type == "ur10":
                    initial_end_effector_orientation = euler_angles_to_quat(np.array([
                        # 0.0, np.pi/2, 0.0 # 35.88706
                        0.0, np.pi/2, np.pi/2 # -54.06432
                    ]))
                elif self._robot_type == "zeus":
                    initial_end_effector_orientation = euler_angles_to_quat(np.array([
                        # 0.0, np.pi, -np.pi/2
                        0.0, np.pi, np.pi/2
                    ]))
                    print(f"initial_end_effector_orientation: {initial_end_effector_orientation}")

            # print(f"initial_picking_position: {initial_picking_position + gripper_to_nut_offset}")

            actions = self._pick_place_controller.forward(
                picking_position=initial_picking_position + gripper_to_nut_offset,
                placing_position=bolt_top + np.array([x_offset, 0.0, 0.0]),
                current_joint_positions=self._robot.get_joint_positions(),
                end_effector_orientation=initial_end_effector_orientation,
            )
            self._robot.apply_action(actions)
            if self._pick_place_controller.is_done():
                self._event = 1

        if self._event == 1:

            if self._robot_type == "zeus":
                # bolt_position = bolt_top + np.array([0.001, 0.005, 0.005])
                bolt_position = bolt_top + np.array([0.0, 0.0, 0.0])
                # bolt_position = bolt_top - np.array([0.0, y_offset, 0.0])
            elif self._robot_type == "ur10":
                bolt_position = bolt_top + np.array([0.0, 0.0, 0.005])
            else:
                bolt_position = bolt_top

            actions2 = self._screw_controller.forward(
                robo_art_controller=self._robot.get_articulation_controller(),
                bolt_position=bolt_position,
                current_joint_positions=self._robot.get_joint_positions(),
                current_joint_velocities=self._robot.get_joint_velocities(),
            )
            self._robot.apply_action(actions2)
            if self._screw_controller.is_paused():
                self.pause()
                self._i += 1
        return

    def reset(self, robot: Robot) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            robot (Robot): Robot
        """
        BaseController.reset(self)
        self._event = 0
        self._pause = False
        self._robot = robot
        self._gripper = self._robot.gripper
        self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.4 / get_stage_units())
        self._pick_place_controller.reset(end_effector_initial_height=self._end_effector_initial_height)
        self._screw_controller.reset()
        return

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return
