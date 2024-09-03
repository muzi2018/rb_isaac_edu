from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.examples.base_sample import BaseSample
from omni.physx.scripts import physicsUtils
import numpy as np
import random
import omni
import carb

class HelloOmniGlass(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)
        self._assets_root_path = get_assets_root_path()
        self.GLASS_URL = self._server_root + "/Projects/RoadBalanceEdu/Wine_Glass.usdz"

        self._num_glass = 0
        self._mtl_prim_list = []
        return

    def add_light(self):
        distantLight1 = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/distantLight1"))
        distantLight1.CreateIntensityAttr(1000)
        distantLight1.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def add_glass_material(self, index):
        mtl_created_list = []
        # Create a new material using OmniGlass.mdl
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created_list,
        )

        omni.kit.commands.execute(
            "MovePrim", 
            path_from="/Looks/OmniGlass", 
            path_to=f"/Looks/OmniGlassScope/OmniGlass{index}",
        )
        mtl_prim = self._stage.GetPrimAtPath(f"/Looks/OmniGlassScope/OmniGlass{index}")

        color_r, color_g, color_b = 1.0, 1.0, 1.0
        color_r, color_g, color_b = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
        glass_depth = random.uniform(0.1, 1.0)
        glass_ior = random.uniform(1.0, 4.0)
        # Set material inputs, these can be determined by looking at the .mdl file
        # or by selecting the Shader attached to the Material in the stage window and looking at the details panel
        omni.usd.create_material_input(mtl_prim, "depth", glass_depth, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mtl_prim, "glass_color", Gf.Vec3f(color_r, color_g, color_b), Sdf.ValueTypeNames.Color3f)
        omni.usd.create_material_input(mtl_prim, "glass_ior", glass_ior, Sdf.ValueTypeNames.Float)
        self._mtl_prim_list.append(mtl_prim)

    def add_wine_glass(self, index, translation):
        x, y, z = translation
        prim_path = f"/World/wine_glass_{index}"
        
        add_reference_to_stage(
            usd_path=self.GLASS_URL, 
            prim_path=prim_path,
        )
        glass_mesh = UsdGeom.Mesh.Get(self._stage, prim_path)
        physicsUtils.set_or_add_translate_op(glass_mesh, Gf.Vec3f(x, y, z))
        physicsUtils.set_or_add_scale_op(glass_mesh, Gf.Vec3f(0.002, 0.002, 0.002))
        physicsUtils.set_or_add_orient_op(glass_mesh, orient=Gf.Quatf(0.5, 0.5, -0.5, -0.5))
        return

    def setup_scene(self):

        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()

        # add light
        self.add_light()

        # add wine glasses & materials
        x_range = np.linspace(-2.0, 2.0, 7)
        y_range = np.linspace(-2.0, 2.0, 7)
        z = 0.05
        
        translations = [
            (x, y, z) for x in x_range for y in y_range
        ]
        self._num_glass = len(translations)

        UsdGeom.Scope.Define(omni.usd.get_context().get_stage(), "/Looks/OmniGlassScope")

        for i, trans in enumerate(translations):
            self.add_wine_glass(i, trans)
            self.add_glass_material(i)

        return

    async def setup_post_load(self):
        self._world = self.get_world()

        for i in range(self._num_glass):
            # Get the path to the prim
            glass_prim = self._stage.GetPrimAtPath(f"/World/wine_glass_{i}")
            # Bind the material to the prim
            glass_mat_shade = UsdShade.Material(self._mtl_prim_list[i])
            UsdShade.MaterialBindingAPI(glass_prim).Bind(
                glass_mat_shade, 
                UsdShade.Tokens.strongerThanDescendants
            )