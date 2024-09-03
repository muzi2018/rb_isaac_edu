from omni.isaac.examples.base_sample import BaseSample
from typing import Optional, Sequence, Tuple 
import numpy as np
import omni
import random

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.objects import DynamicCuboid
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.physx.scripts import physicsUtils, particleUtils, deformableUtils

class HelloMDLMaterial(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._assets_root_path = get_assets_root_path()
        self.CUBE_URL = self._assets_root_path + "/Isaac/Props/Shapes/cube.usd"
        return

    def add_mdl_material(self):
        success, result = omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="omniverse://localhost/NVIDIA/Materials/Base/Masonry/Concrete_Smooth.mdl",
            mtl_name="Concrete_Smooth",
            mtl_path="/World/Looks/Concrete_Smooth"
        )
        concrete_material = UsdShade.Material(self._stage.GetPrimAtPath("/World/Looks/Concrete_Smooth"))

        success, result = omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="omniverse://localhost/NVIDIA/Materials/Base/Metals/Brass.mdl",
            mtl_name="Brass",
            mtl_path="/World/Looks/Brass"
        )

        brass_material = UsdShade.Material(self._stage.GetPrimAtPath("/World/Looks/Brass"))
        return concrete_material, brass_material

    def add_cube(self):
        add_reference_to_stage(
            usd_path=self.CUBE_URL, 
            prim_path=f"/World/Cube",
        )
        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(0, 0, 0.5))
        physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(0.5, 0.5, 0.5))
        return

    def setup_scene(self):

        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()

        self.add_cube()
        self._concrete_material, self._brass_material = self.add_mdl_material()
        return

    async def setup_post_load(self):
        self._world = self.get_world()

        floor_prim = self._stage.GetPrimAtPath("/World/defaultGroundPlane/Enviroment/Geometry")
        UsdShade.MaterialBindingAPI(floor_prim).Bind(
            self._concrete_material, 
            UsdShade.Tokens.strongerThanDescendants
        )

        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Cube/Cube")
        UsdShade.MaterialBindingAPI(cube_mesh).Bind(
            self._brass_material, 
            UsdShade.Tokens.strongerThanDescendants
        )
        return
