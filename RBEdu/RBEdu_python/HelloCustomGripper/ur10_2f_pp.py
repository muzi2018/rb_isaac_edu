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

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.nucleus import get_url_root
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core import SimulationContext

import numpy as np
import carb

from omni.isaac.custom_ur10 import UR102F, PickPlaceController


class UR10CustomGripper(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # Nucleus Path Configuration
        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)
        self.UR_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR10/ur10_2f85.usd"
        
        self._sim_count = 0
        self._cube_position = np.array([0.5, -0.5, 0.03])
        return

    def _setup_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_solver_type("TGS")
        self._scene.set_broadphase_type("GPU")

    def add_ur10(self):
        # Add UR10 robot
        self._ur10 = self._world.scene.add(
            UR102F(
                prim_path="/World/UR10",
                name="UR10",
                usd_path=self.UR_PATH,
                attach_gripper=True,
            )
        )

    def add_cube(self):
        self.fancy_cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/fancy_cube",
                name="fancy_cube",
                position=self._cube_position,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0, 0, 1.0]),
            ))

    def setup_scene(self):

        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.add_cube()
        self.add_ur10()

        self._setup_simulation()
        self.simulation_context = SimulationContext()
        return

    async def setup_post_load(self):
        self._robot = self._world.scene.get_object("UR10")
        self._gripper = self._robot.gripper

        # Pick and Place Controller
        self._pap_controller = PickPlaceController(
            name="pick_place_controller", 
            gripper=self._robot.gripper, 
            robot_articulation=self._robot,
            fast_movement=False,
        )

        self._articulation_controller = self._robot.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    def physics_step(self, step_size):

        self._sim_count += 1

        joints_state = self._robot.get_joints_state()
        actions = self._pap_controller.forward(
            picking_position=self._cube_position,
            placing_position=np.array([0.5, 0.5, 0.03]),
            current_joint_positions=joints_state.positions,
            end_effector_offset=np.array([-0.15, 0.0, 0.01]),
            end_effector_orientation=np.array([ 
                # 0.7071068, 0, 0.7071068, 0, 
                # 0.7071068, 0.7071068, 0, 0, 
                # 1, 0, 0, 0, 
                # 0.7071068, 0, 0.7071068, 0, 
                0.9238795, 0, 0, 0.3826834, 
            ]),
        )
        if self._pap_controller.is_done():
            print("done picking and placing")
        else:
            self._articulation_controller.apply_action(actions)
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        await self._world.play_async()
        return
    
    def world_cleanup(self):
        return