from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.examples.base_sample import BaseSample
from omni.physx.scripts import physicsUtils
import omni


class HelloOmniGlass(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._mtl_prim = None
        self._assets_root_path = get_assets_root_path()
        self.CUBE_URL = self._assets_root_path + "/Isaac/Props/Shapes/cube.usd"
        return

    def add_glass_material(self):
        mtl_created_list = []
        # Create a new material using OmniGlass.mdl
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created_list,
        )
        self._mtl_prim = self._stage.GetPrimAtPath(mtl_created_list[0])

        # Set material inputs, these can be determined by looking at the .mdl file
        # or by selecting the Shader attached to the Material in the stage window and looking at the details panel
        omni.usd.create_material_input(self._mtl_prim, "depth", 0.5, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(self._mtl_prim, "glass_color", Gf.Vec3f(1, 1, 1), Sdf.ValueTypeNames.Color3f)
        omni.usd.create_material_input(self._mtl_prim, "glass_ior", 3.0, Sdf.ValueTypeNames.Float)

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
        self.add_glass_material()
        self.add_glass_material()
        return

    async def setup_post_load(self):
        self._world = self.get_world()

        # Get the path to the prim
        cube_prim = self._stage.GetPrimAtPath("/World/Cube/Cube")
        # Bind the material to the prim
        cube_mat_shade = UsdShade.Material(self._mtl_prim)
        UsdShade.MaterialBindingAPI(cube_prim).Bind(
            cube_mat_shade, 
            UsdShade.Tokens.strongerThanDescendants
        )