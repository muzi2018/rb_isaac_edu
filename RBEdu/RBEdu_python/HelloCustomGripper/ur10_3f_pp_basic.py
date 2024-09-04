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

from omni.isaac.universal_robots.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.nucleus import get_url_root
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder

from omni.isaac.custom_ur10 import UR103F, PickPlaceController
from omni.isaac.core import SimulationContext

import numpy as np
import carb


class UR10CustomGripper(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # Nucleus Path Configuration
        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        self._sim_count = 0
        self._gripper_mode = "basic" # "basic" or "wide" or "pinch"
        self._obj_position = np.array([0.5, -0.5, 0.07])
        return

    def add_ur10(self):
        # Add UR10 robot
        if self._gripper_mode == "basic":
            self.UR_PATH = "omniverse://localhost/Projects/ETRI/Manipulator/UR10/ur10_3f_grasping.usd"
        else:
            self.UR_PATH = "omniverse://localhost/Projects/ETRI/Manipulator/UR10/ur10_3f_picking.usd"
        
        self._ur10 = self._world.scene.add(
            UR103F(
                prim_path="/World/UR10",
                name="UR10",
                position=np.array([0.0, 0.0, 0.0]),
                usd_path=self.UR_PATH,
                attach_gripper=True,
                gripper_mode=self._gripper_mode, # basic, pinch, wide, scissor
            )
        )

    def add_obj(self):
        self.fancy_cylinder = self._world.scene.add(
            DynamicCylinder(
                prim_path="/World/fancy_cylinder",
                name="fancy_cylinder",
                position=self._obj_position,
                scale=np.array([0.05, 0.05, 0.1]),
                color=np.array([0, 0, 1.0]),
            ))

    def setup_scene(self):

        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.add_ur10()
        self.add_obj()

        self.simulation_context = SimulationContext()
        return

    async def setup_post_load(self):
        self._my_ur10 = self._world.scene.get_object("UR10")
        self._my_gripper = self._my_ur10.gripper

        if self._gripper_mode == "pinch":
            subset_names = ['palm_finger_1_joint', 'palm_finger_2_joint'] 
            robot_subset = ArticulationSubset(self._my_ur10, subset_names)
            robot_subset.apply_action([-10.0, 10.0])
        elif self._gripper_mode == "wide":
            subset_names = ['palm_finger_1_joint', 'palm_finger_2_joint'] 
            robot_subset = ArticulationSubset(self._my_ur10, subset_names)
            robot_subset.apply_action([10.0, -10.0])

        # Pick and Place Controller
        self._pap_controller = PickPlaceController(
            name="pick_place_controller", 
            gripper=self._my_ur10.gripper, 
            robot_articulation=self._my_ur10,
            end_effector_initial_height=0.5,
        )

        self._articulation_controller = self._my_ur10.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    def physics_step(self, step_size):

        self._sim_count += 1

        joints_state = self._my_ur10.get_joints_state()
        actions = self._pap_controller.forward(
            picking_position=self._obj_position,
            placing_position=np.array([0.5, 0.5, 0.1]),
            current_joint_positions=joints_state.positions,
            end_effector_offset=np.array([-0.16, 0.0, 0.0]),
            end_effector_orientation=np.array([
                1, 0, 0, 0,
            ])
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