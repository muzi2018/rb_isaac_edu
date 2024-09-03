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

from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.physx.scripts import physicsUtils  

from omni.isaac.examples.base_sample import BaseSample
from pxr import UsdGeom, Gf

from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import Franka

import numpy as np
import omni


#### Example 5 - Physics Material
class HelloManip(BaseSample):

    def __init__(self) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()
        self.CUBE_URL = self._isaac_assets_path + "/Isaac/Props/Blocks/nvidia_cube.usd"
        
        self._sim_count = 0
        return
    
    def add_robot(self):
        self._franka = self._world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="Franka",
            )
        )
        return

    def add_cube(self):
        self._cube = add_reference_to_stage(usd_path=self.CUBE_URL, prim_path=f"/World/NVIDIA_Cube")
        self._stage = omni.usd.get_context().get_stage()

        # modify cube mesh
        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/NVIDIA_Cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.3, -0.3, 0.1))

        # # modify cube mass
        # cube_mass = UsdPhysics.MassAPI.Apply(get_current_stage().GetPrimAtPath("/World/NVIDIA_Cube"))
        # cube_mass.CreateMassAttr().Set(0.1)
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self.add_robot()
        self.add_cube()
        return

    async def modify_cube_friction(self):
        self.high_friction_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/high_friction_material",
            name="high_friction_material",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        await self._world.reset_async()

        # modify cube material
        geometry_prim = GeometryPrim(prim_path="/World/NVIDIA_Cube/S_NvidiaCube")
        geometry_prim.apply_physics_material(self.high_friction_material)
        await self._world.play_async()
        return
    
    async def setup_post_load(self):
        self._world = self.get_world()

        # await self.modify_cube_friction()

        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

        self._franka.gripper.open()
        self._articulation_controller = self._franka.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    def physics_step(self, step_size):

        self._sim_count += 1

        joints_state = self._franka.get_joints_state()
        actions = self._controller.forward(
            picking_position=np.array([0.3, -0.3, 0.03]),
            placing_position=np.array([0.3, 0.3, 0.05]),
            current_joint_positions=joints_state.positions,
        )

        if self._controller.is_done():
            self._world.pause()
        else:
            self._articulation_controller.apply_action(actions)
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._controller.reset()
        return

    def world_cleanup(self):
        return
