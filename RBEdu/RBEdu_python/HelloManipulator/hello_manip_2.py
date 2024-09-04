# Copyright 2024 Road Balance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka

import numpy as np

#### Example 2 - Pick and Place
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

        self._franka = world.scene.add(
            Franka(
                prim_path="/World/Fancy_Franka",
                name="fancy_franka"
            )
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")

        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):

        current_joint_positions = self._franka.get_joint_positions()

        actions = self._controller.forward(
            picking_position=np.array([0.3, 0.3, 0.02]),
            placing_position=np.array([0.3, -0.3, 0.02]),
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return