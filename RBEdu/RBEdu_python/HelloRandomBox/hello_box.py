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
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom,Gf, UsdLux

import omni.usd
import random


class HelloBox(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def add_light(self):
        # Turn off the default light
        prim = self._stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
        attribute = prim.GetAttribute("intensity")
        attribute.Set(0)

        # add_dome_light
        light = UsdLux.DomeLight.Define(self._stage, "/World/dome_light")

        # intensity
        light.CreateIntensityAttr().Set(2000)
        shaping = UsdLux.ShapingAPI(light)
        shaping.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr().Set(180)
        shaping.CreateShapingConeSoftnessAttr()
        shaping.CreateShapingFocusAttr()
        shaping.CreateShapingFocusTintAttr()
        shaping.CreateShapingIesFileAttr()

    def add_random_size_boxes(self):
        # Create a grid of boxes
        all_boxes = self._stage.DefinePrim("/World/boxes", "Xform")
        for i in range(5):
            for j in range(5):
                path = f"/World/boxes/box_{i}_{j}"
                # Add box of random size
                size = (
                    random.uniform(0.2, 1.0),
                    random.uniform(0.2, 1.0),
                    random.uniform(0.2, 1.0),
                )
                translate = (5*i, 5*j, 0)
                cube_prim = self._stage.DefinePrim(path, "Cube")
                cube_mesh = UsdGeom.Mesh.Get(self._stage, path)
                physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(translate))
                physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(size))
                print(f"Adding box at {path} with size {size} and translate {translate}")

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()

        self.add_light()
        self.add_random_size_boxes()
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        return
