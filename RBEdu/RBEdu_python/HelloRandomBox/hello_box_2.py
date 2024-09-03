from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.examples.base_sample import BaseSample
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom,Gf, UsdLux

import omni.replicator.core as rep
import numpy as np
import omni.usd
import random

class HelloBox(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()   # omniverse://localhost/NVIDIA/Assets/Isaac/4.0
        self.Cardbox_C1_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_C1.usd"
        self.Cardbox_A3_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A3.usd"
        self.Cardbox_A2_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A2.usd"
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

    def add_random_cardbox(self):
        # Create a grid of boxes
        self._cardbox = rep.create.from_usd(self.Cardbox_A3_URL)

    def setup_scene(self):

        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()

        self.add_light()
        self.add_random_cardbox()
        return

    async def setup_post_load(self):
        self._world = self.get_world()

        with rep.trigger.on_frame(num_frames=20):
            with self._cardbox:
                rep.modify.pose(
                    position=rep.distribution.uniform((-0.1, -0.1, 1.0), (0.1, 0.1, 1.0)),
                    rotation=rep.distribution.uniform((-360,-360, -360), (360, 360, 360)),
                    scale=0.01,
                )

        return
