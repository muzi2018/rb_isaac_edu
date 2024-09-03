
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask
from omni.isaac.franka import Franka

import numpy as np

class FrankaPlaying(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([0.3, -0.3, 0.02])
        self._task_achieved = False
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0, 0, 1.0])))
        self._franka = scene.add(Franka(prim_path="/World/Fancy_Franka",
                                        name="fancy_franka"))
        return

    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations

    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return


class HelloManip(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.add_task(FrankaPlaying(name="my_first_task"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        current_observations = self._world.get_observations()

        actions = self._controller.forward(
            picking_position=current_observations["fancy_cube"]["position"],
            placing_position=current_observations["fancy_cube"]["goal_position"],
            current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
        )

        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return