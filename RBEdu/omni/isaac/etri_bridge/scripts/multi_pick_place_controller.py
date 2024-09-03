from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.robots.robot import Robot
import numpy as np
import typing

from .custom_pick_place_controller import PrevCustomPickPlaceController


class MultiPickPlaceController(BaseController):
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
            cspace_controller: BaseController,
            end_effector_initial_height: None,
        ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._robot = robot
        self._gripper = self._robot.gripper

        if end_effector_initial_height is None:
            self._end_effector_initial_height = 1.2
        # self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.4 / get_stage_units())
        self._end_effector_initial_height = end_effector_initial_height

        self._pause = False
        self._cspace_controller = cspace_controller

        pick_place_events_dt = [0.005, 0.01, 0.2, 0.2, 0.01, 0.005, 0.002, 0.05, 0.008, 0.01]
        
        self._pick_place_controller = PrevCustomPickPlaceController(
            name="pickplace_controller",
            cspace_controller=self._cspace_controller,
            gripper=self._gripper,
            end_effector_initial_height=self._end_effector_initial_height,
            events_dt=pick_place_events_dt,
        )
        
        self._motion_dict = None
        self._is_done = False
        self._i = 0
        return
    
    def is_done(self):
        return self._is_done

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

    def get_pp_event(self):
        return self._pick_place_controller.get_current_event()

    def set_motion_dict(
            self,
            motion_dict: dict,
        ):
        self._motion_dict = motion_dict
        self._object_names = list(self._motion_dict.keys())[1:]
        self._event_len = len(self._object_names)
        self._end_effector_offset = motion_dict["end_effector_offset"]

    def set_world(
            self,
            world,
        ):
        self._world = world

    def get_observation(self, object_name):
        object_geom = self._world.scene.get_object(object_name)
        object_position, object_orientation = object_geom.get_world_pose()
        return object_position, object_orientation

    def forward(self):
        """Runs the controller one step.

        Args:
            initial_picking_position (np.ndarray): initial nut position at table feeder
            bolt_top (np.ndarray):  bolt target position

        #"""

        print(f"MultiPickPlaceController event: {self._event}")

        object_name = self._object_names[self._event]
        object_position, _ = self.get_observation(object_name)
        picking_offset = self._motion_dict[object_name]["picking_offset"]
        picking_object_offset = self._motion_dict[object_name]["picking_object_offset"]
        picking_orientation = self._motion_dict[object_name]["picking_orientation"]

        placing_position = self._motion_dict[object_name]["placing_position"]
        placing_offset = self._motion_dict[object_name]["placing_offset"]
        placing_object_offset = self._motion_dict[object_name]["placing_object_offset"]
        placing_orientation = self._motion_dict[object_name]["placing_orientation"]

        picking_position = object_position + picking_object_offset
        placing_position = placing_position + placing_object_offset

        actions = self._pick_place_controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=self._robot.get_joint_positions(),
            picking_offset=picking_offset,
            placing_offset=placing_offset,
            end_effector_offset=self._end_effector_offset,
            picking_end_effector_orientation=picking_orientation,
            placing_end_effector_orientation=placing_orientation,
        )

        if self._pick_place_controller.is_done():
            print("self._pick_place_controller.is_done()")
            self._pick_place_controller.reset()
            self._event += 1
            if self._event == self._event_len:
                self._is_done = True
                self._event = 0

        return actions

    def reset(self, robot: Robot) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            robot (Robot): Robot
        """
        BaseController.reset(self)
        self._event = 0
        self._pause = False
        self._is_done = False
        self._robot = robot
        self._gripper = self._robot.gripper
        # self._end_effector_initial_height = self._robot.get_world_pose()[0][2] + (0.4 / get_stage_units())
        # self._end_effector_initial_height = 1.2
        self._pick_place_controller.reset(end_effector_initial_height=self._end_effector_initial_height)
        return

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return
