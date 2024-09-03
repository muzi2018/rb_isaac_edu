from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.core.articulations import Articulation
import numpy as np
import typing

from .custom_pick_place_controller import CustomPickPlaceController


class MultiPickPlaceController:

    def __init__(
            self,
            name: str,
            gripper: Gripper,
            robot_articulation: Articulation,
            end_effector_initial_height: typing.Optional[float] = None,
            robot_type: str = "franka",
            events_dt: typing.Optional[typing.List[float]] = None,
        ) -> None:

        self._controller = CustomPickPlaceController(
            name=name,
            gripper=gripper,
            robot_articulation=robot_articulation,
            end_effector_initial_height=end_effector_initial_height,
            robot_type=robot_type,
        )

        self._event = 0
        self._event_len = 0
        self._is_done = False
        self._motion_dict = None
        self._object_names = []
        self._pick_place_controllers = None

        return
    
    def reset(self):
        self._event = 0
        self._is_done = False
        return

    def is_done(self):
        return self._is_done

    def set_world(
            self,
            world,
        ):
        self._world = world

    def set_pick_place_controller(
            self,
            pick_place_controllers: CustomPickPlaceController,
        ):
        self._pick_place_controllers = pick_place_controllers

    def set_end_effector_offset(
            self,
            end_effector_offset: np.ndarray,
        ):
        self._end_effector_offset = end_effector_offset

    def set_motion_dict(
            self,
            motion_dict: dict,
        ):
        self._motion_dict = motion_dict
        self._object_names = list(self._motion_dict.keys())
        self._event_len = len(self._object_names)

    def get_observation(self, object_name):
        object_geom = self._world.scene.get_object(object_name)
        object_position, object_orientation = object_geom.get_world_pose()
        return object_position, object_orientation

    def get_event(self):
        return self._event

    def get_pp_event(self):
        return self._pick_place_controllers.get_current_event()

    def forward(
            self,
            current_joint_positions: np.ndarray,
        ):

        print(f"MultiPickPlaceController event: {self._event}")

        object_name = self._object_names[self._event]
        object_position, object_orientation = self.get_observation(object_name)
        picking_offset = self._motion_dict[object_name]["picking_offset"]
        picking_position_offset = self._motion_dict[object_name]["picking_position_offset"]
        picking_orientation_offset = self._motion_dict[object_name]["picking_orientation_offset"]

        placing_bin_name = self._motion_dict[object_name]["placing_bin_name"]
        bin_position, bin_orientation = self.get_observation(placing_bin_name)
        placing_offset = self._motion_dict[object_name]["placing_offset"]
        placing_position_offset = self._motion_dict[object_name]["placing_position_offset"]
        placing_orientation_offset = self._motion_dict[object_name]["placing_orientation_offset"]

        picking_position = object_position + picking_position_offset
        placing_position = bin_position + placing_position_offset

        actions = self._pick_place_controllers.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            end_effector_orientation=picking_orientation_offset,
            placing_end_effector_orientation=placing_orientation_offset,
            picking_offset=picking_offset,
            placing_offset=placing_offset,
            end_effector_offset=self._end_effector_offset,
            current_joint_positions=current_joint_positions,
        )

        if self._pick_place_controllers.is_done():
            print("self._pick_place_controllers.is_done()")
            self._pick_place_controllers.reset()
            self._event += 1
            if self._event == self._event_len:
                self._is_done = True
                self._event = 0

        return actions
