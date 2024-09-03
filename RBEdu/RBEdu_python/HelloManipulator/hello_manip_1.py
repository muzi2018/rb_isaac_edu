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

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka

import numpy as np

#### Example 1 - Gripper Control
class HelloManip(BaseSample):

    def __init__(self) -> None:
        super().__init__()
        self._sim_count = 0
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self._franka = world.scene.add(
            Franka(
                prim_path="/World/Fancy_Franka",
                name="fancy_franka"
            )
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):

        if self._sim_count % 100 == 0:
            self._franka.gripper.set_joint_positions(self._franka.gripper.joint_closed_positions)
        elif self._sim_count % 50 == 0:
            self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)

        self._sim_count += 1
        return