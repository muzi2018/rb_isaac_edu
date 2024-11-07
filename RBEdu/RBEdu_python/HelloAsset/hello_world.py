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

from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.nucleus import get_assets_root_path, get_url_root
from omni.physx.scripts import deformableUtils, physicsUtils  

from omni.isaac.examples.base_sample import BaseSample
from pxr import UsdGeom, Gf

import omni
import carb


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()

        self.CUBE_URL = self._isaac_assets_path + "/Isaac/Props/Blocks/nvidia_cube.usd"
        self.HYDRANT_PATH = "omniverse://localhost/Projects/RoadBalanceEdu/Hydrant.usdz"
        omniverse://localhost/Projects/RoadBalanceEdu/Hydrant.usdz
        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()

        add_reference_to_stage(usd_path=self.CUBE_URL, prim_path=f"/World/NVIDIA_Cube")
        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/NVIDIA_Cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.5))
        physicsUtils.set_or_add_orient_op(cube_mesh, orient=Gf.Quatf(0.9238795, 0.3826834, 0, 0))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(1.2, 1.2, 1.2))

        add_reference_to_stage(usd_path=self.HYDRANT_PATH, prim_path=f"/World/Hydrant")
        hydrant_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Hydrant")
        physicsUtils.set_or_add_translate_op(hydrant_mesh, translate=Gf.Vec3f(0.0, 0.2, 0.5))
        physicsUtils.set_or_add_orient_op(hydrant_mesh, orient=Gf.Quatf(0.9238795, 0, 0.3826834, 0))
        physicsUtils.set_or_add_scale_op(hydrant_mesh, scale=Gf.Vec3f(0.001, 0.001, 0.001))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
