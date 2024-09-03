from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

import numpy as np
import typing

class ScrewController(BaseController):
    """
    A state machine for screwing nuts on bolts

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - State 0: Lower end_effector down to encircle the nut
    - State 1: Close grip
    - State 2: Re-Center end-effector grip with that of the nut and bolt
    - State 3: Screw Clockwise
    - State 4: Open grip (initiates at this state and cycles until limit)
    - State 5: Screw counter-clockwise

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 5 or less
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        robot_type: str = "franka",
    ) -> None:
        BaseController.__init__(self, name=name)
        # Here 4 
        self._event = 4
        self._t = 0
        self._events_dt = events_dt
        self._robot_type = robot_type

        if self._events_dt is None:
            if self._robot_type == "franka" or self._robot_type == "franka_normal":
                self._events_dt = [0.01, 0.1, 0.05, 0.025, 0.05, 0.05]
            elif self._robot_type == "ur5":
                self._events_dt = [0.015, 0.1, 0.1, 0.017, 0.08, 0.018]
            elif self._robot_type == "ur10":
                self._events_dt = [0.015, 0.1, 0.1, 0.014, 0.08, 0.015]
            elif self._robot_type == "zeus":
                self._events_dt = [0.015, 0.05, 0.1, 0.015, 0.05, 0.012]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 5:
                raise Exception("events dt need have length of 5 or less")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        self._screw_position = np.array([0.0, 0.0, 0.0])
        self._final_position = np.array([0.0, 0.0, 0.0])
        self._screw_speed = 360.0 / 180.0 * np.pi
        self._screw_speed_back = 720.0 / 180.0 * np.pi
        
        if self._robot_type == "franka" or self._robot_type == "franka_normal":
            self._wrist_joint_idx = 6
        elif self._robot_type == "ur5":
            self._wrist_joint_idx = 5
        elif self._robot_type == "ur10":
            self._wrist_joint_idx = 5
        elif self._robot_type == "zeus":
            self._wrist_joint_idx = 5

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
        robo_art_controller: ArticulationController,
        bolt_position: np.ndarray,
        current_joint_positions: np.ndarray,
        current_joint_velocities: np.ndarray,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            robo_art_controller (ArticulationController): Robot's Articulation Controller.
            bolt_position (np.ndarray): bolt position to reference for screwing position.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            current_joint_velocities (np.ndarray): Current joint velocities of the robot.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """

        print(f"[ScrewController] event : {self._event}")
        # print(f"current_joint_positions: {current_joint_positions}")

        if self._pause or self._event >= len(self._events_dt):
            target_joints = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joints)

        if self._event == 0 and self._start:
            self._screw_position = np.copy(bolt_position)
            self._final_position = np.copy(bolt_position)
            self._start = False
            self._target_end_effector_orientation = self._gripper.get_world_pose()[1]

        if self._event == 0:
            robo_art_controller.switch_dof_control_mode(dof_index=self._wrist_joint_idx, mode="position")
            orientation_quat = self._gripper.get_world_pose()[1]
            self.orientation_euler = quat_to_euler_angles(orientation_quat)
            target_orientation_euler = np.array([
                self.orientation_euler[0], self.orientation_euler[1], -np.pi / 2
            ])

            target_orientation_quat = euler_angles_to_quat(target_orientation_euler)
            # print(f"self.orientation_euler : {self.orientation_euler}")
            # print(f"target_orientation_euler : {target_orientation_euler}")

            if self._robot_type == "franka" or self._robot_type == "franka_normal":
                finger_pos = current_joint_positions[-2:]
                positive_x_offset = finger_pos[1] - finger_pos[0]
            elif self._robot_type == "ur5":
                positive_x_offset = 0.0
            elif self._robot_type == "ur10":
                positive_x_offset = -0.001
            elif self._robot_type == "zeus":
                positive_x_offset = 0.0

            if self._robot_type == "ur10":
                target_joints = self._cspace_controller.forward(
                    target_end_effector_position=self._screw_position + np.array([positive_x_offset, 0.0, 0.0]),
                    target_end_effector_orientation=orientation_quat,
                )
            elif self._robot_type == "franka_normal":
                target_joints = self._cspace_controller.forward(
                    target_end_effector_position=self._screw_position + np.array([positive_x_offset, 0.0, -0.001]),
                    target_end_effector_orientation=target_orientation_quat,
                )
            else:
                target_joints = self._cspace_controller.forward(
                    target_end_effector_position=self._screw_position,
                    target_end_effector_orientation=target_orientation_quat,
                )

        if self._event == 1:
            self._lower = False
            robo_art_controller.switch_dof_control_mode(dof_index=self._wrist_joint_idx, mode="position")
            target_joints = self._gripper.forward(action="close")

        if self._event == 2:
            robo_art_controller.switch_dof_control_mode(dof_index=self._wrist_joint_idx, mode="position")
            orientation_quat = self._gripper.get_world_pose()[1]
            self.orientation_euler = quat_to_euler_angles(orientation_quat)
            target_orientation_euler = np.array([self.orientation_euler[0], self.orientation_euler[1], -np.pi / 2])
            target_orientation_quat = euler_angles_to_quat(target_orientation_euler)
            
            if self._robot_type == "franka":
                positive_x_offset = 0.0
            elif self._robot_type == "franka_normal":
                finger_pos = current_joint_positions[-2:]
                positive_x_offset = finger_pos[1] - finger_pos[0]
            elif self._robot_type == "ur5":
                positive_x_offset = 0.0
            elif self._robot_type == "ur10":
                positive_x_offset = -0.001
                target_orientation_quat = orientation_quat
            elif self._robot_type == "zeus":
                positive_x_offset = 0.0
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=self._screw_position + np.array([positive_x_offset, 0.0, -0.001]),
                # target_end_effector_position=self._screw_position + np.array([positive_x_offset, 0.0, 0.0]),
                target_end_effector_orientation=target_orientation_quat,
            )

        if self._event == 3:
            robo_art_controller.switch_dof_control_mode(dof_index=self._wrist_joint_idx, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[self._wrist_joint_idx] = self._screw_speed
            print(f"[E3] current_joint_positions: {current_joint_positions[self._wrist_joint_idx]}")
            
            if self._robot_type == "franka" and current_joint_positions[self._wrist_joint_idx] > 2.7:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "franka_normal" and current_joint_positions[self._wrist_joint_idx] > 2.7:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "ur5" and current_joint_positions[self._wrist_joint_idx] > 0.05:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "ur10" and current_joint_positions[self._wrist_joint_idx] > 0.05:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "zeus" and current_joint_positions[self._wrist_joint_idx] > 1.7:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        if self._event == 4:
            # # Here
            # if self._start:
            #     # target_joints = [
            #     #     1.7479901e+00, 4.3235725e-01, 1.2947322e+00, 5.8915528e-05, 1.4143754e+00,
            #     #     1.7480165e+00, 2.2557667e-09, 5.7413474e-09, 2.7053468e-06, 1.7313969e-06,
            #     #     2.6218695e-06, 6.8967506e-07 
            #     # ]
            #     target_joints = [
            #         3.2631746e-01, 6.3756488e-02, 3.1738363e-02, -1.8287961e+00, 
            #         -2.0849844e-03, 1.8885007e+00, 2.7144048e+00, 3.9999999e-02, 
            #         3.9999999e-02
            #     ]

            #     target_joints = ArticulationAction(joint_positions=target_joints)
            # else:
            robo_art_controller.switch_dof_control_mode(dof_index=self._wrist_joint_idx, mode="position")
            target_joints = self._gripper.forward(action="open")

        if self._event == 5:
            robo_art_controller.switch_dof_control_mode(dof_index=self._wrist_joint_idx, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[self._wrist_joint_idx] = -self._screw_speed_back
            print(f"[E5] current_joint_positions: {current_joint_positions[self._wrist_joint_idx]}")

            # angle decrease
            if self._robot_type == "zeus" and current_joint_positions[self._wrist_joint_idx] < -1.4:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "franka" and current_joint_positions[self._wrist_joint_idx] < -0.35:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "franka_normal" and current_joint_positions[self._wrist_joint_idx] < -0.35:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            elif self._robot_type == "ur10" and current_joint_positions[self._wrist_joint_idx] < -3.1:
                target_joint_velocities[self._wrist_joint_idx] = 0.0
            
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event = (self._event + 1) % 6
            self._t = 0
            if self._event == 5:
                if not self._start and (bolt_position[2] - self._final_position[2] > 0.0198):
                    self.pause()
                    return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
                if self._start:
                    self._screw_position[2] -= 0.001
                    self._final_position[2] -= 0.001
                if bolt_position[2] - self._screw_position[2] < 0.013:
                    self._screw_position[2] -= 0.0018
                self._final_position[2] -= 0.0018

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 5 or less
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 4
        self._t = 0
        self._pause = False
        self._start = True
        self._screw_position = np.array([0.0, 0.0, 0.0])
        self._final_position = np.array([0.0, 0.0, 0.0])
        self._screw_speed = 360.0 / 180.0 * np.pi
        self._screw_speed_back = 720.0 / 180.0 * np.pi
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 5:
                raise Exception("events dt need have length of 5 or less")
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
