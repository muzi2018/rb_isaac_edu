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

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid

import numpy as np
import omni.usd


class HelloLight(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # Turn off the default light
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
        attribute = prim.GetAttribute("intensity")
        attribute.Set(0)

        fancy_cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube", # The prim path of the cube in the USD stage
                name="fancy_cube", # The unique name used to retrieve the object from the scene later on
                position=np.array([0, 0, 1.0]), # Using the current stage units which is in meters by default.
                scale=np.array([0.5015, 0.5015, 0.5015]), # most arguments accept mainly numpy arrays.
                color=np.array([0, 0, 1.0]), # RGB channels, going from 0-1
            ))
        
        # TODO: Edit SphereLight Properties
        sphereLight = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/SpecialSphereLight"))
        sphereLight.CreateRadiusAttr(0.5)
        sphereLight.CreateIntensityAttr(1000000.0)
        sphereLight.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 3.0))
        sphereLight.AddRotateXYZOp().Set((0, 0, 0))
        sphereLight.CreateColorAttr(Gf.Vec3f(0.1456, 0.4883, 0.7341))
        shaping = UsdLux.ShapingAPI(sphereLight)
        shaping.CreateShapingConeAngleAttr().Set(45)
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        return