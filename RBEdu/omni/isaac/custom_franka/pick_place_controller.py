import typing
from typing import List, Optional

from omni.isaac.core.articulations import Articulation
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController


class PickPlaceController(manipulators_controllers.PickPlaceController):
# class PickPlaceController():
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
    ) -> None:
        if events_dt is None:
            events_dt = [0.01, 0.0035, 0.01, 1.0, 0.008, 0.005, 0.005, 1, 0.01, 0.08]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", 
                robot_articulation=robot_articulation, 
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        return
