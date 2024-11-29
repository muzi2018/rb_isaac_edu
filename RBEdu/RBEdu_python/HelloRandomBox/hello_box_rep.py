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
import omni.replicator.core as rep
import omni.usd

from pxr import UsdGeom,Gf, UsdLux


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

    def color_randomize(self):
        with self.red_cube:
            rep.randomizer.materials(self.cube_mat)
        return self.red_cube.node

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()

        self.add_light()
        self._cube = rep.create.cube(
            semantics=[('class', 'cube')],  
            position=rep.distribution.uniform((0, 0, 0), (25, 25, 0)),
            scale=rep.distribution.uniform((0.2, 0.2, 0.2), (1.0, 1.0, 1.0)),
            count=25
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        return
