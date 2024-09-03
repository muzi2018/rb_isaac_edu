from omni.isaac.universal_robots.controllers import PickPlaceController
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask
from omni.isaac.universal_robots import UR10

import numpy as np

class HelloManip(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._sim_count = 0
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self._cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0.3, 0.3, 0.3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 0, 1.0])
            )
        )

        self._ur10 = world.scene.add(
            UR10(
                prim_path="/World/UR10",
                name="ur10",
                attach_gripper=True,
                position=np.array([0.0, 0.0, 0.3])
            )
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._ur10 = self._world.scene.get_object("ur10")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._ur10.gripper,
            robot_articulation=self._ur10,
        )

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):

        current_joint_positions = self._ur10.get_joint_positions()
        # cube_position, _ = self._cube.get_world_pose()

        actions = self._controller.forward(
            # picking_position=cube_position,
            picking_position=np.array([0.3, 0.3, 0.02]),
            placing_position=np.array([0.3, -0.3, 0.02]),
            current_joint_positions=current_joint_positions,
        )
        self._ur10.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()

        return