from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.examples.base_sample import BaseSample
from semantics.schema.editor import PrimSemanticData
from omni.physx.scripts import physicsUtils  

import omni.replicator.core as rep
import random
import omni
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


def add_object(
        usd_path: str, 
        prim_path: str,
        translate: tuple = (0.0, 0.0, 0.0),
        orient: tuple = (0.0, 0.0, 0.0, 1.0),
        scale: tuple = (1.0, 1.0, 1.0),
    ) -> None:
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    obj_mesh = UsdGeom.Mesh.Get(get_current_stage(), prim_path)
    tx, ty, tz = tuple(translate)
    physicsUtils.set_or_add_translate_op(
        obj_mesh,
        translate=Gf.Vec3f(tx, ty, tz),
    )
    rw, rx, ry, rz = tuple(orient)
    physicsUtils.set_or_add_orient_op(
        obj_mesh,
        orient=Gf.Quatf(rw, rx, ry, rz),
    )

    # if scale type if tuple
    if isinstance(scale, tuple):
        sx, sy, sz = tuple(scale)
        physicsUtils.set_or_add_scale_op(
            obj_mesh,
            scale=Gf.Vec3f(sx, sy, sz),
        )
    elif isinstance(scale, float):
        physicsUtils.set_or_add_scale_op(
            obj_mesh,
            scale=Gf.Vec3f(scale, scale, scale),
        )

def add_semantic_data(
        prim_path: str, 
        class_name: str
    ) -> None:
    prim = get_current_stage().GetPrimAtPath(prim_path)
    prim_sd = PrimSemanticData(prim)
    prim_sd.add_entry("class", class_name)

def get_textures(dir_path, png_type=".png"):
    textures = []
    dir_path += "/"
    for file in os.listdir(dir_path):
        if file.endswith(png_type):
            textures.append(dir_path + file)
    return textures


class DefectEnv(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._sim_count = 0

        self._num_scratch_textures = 0
        # self._target_prim = "/World/Car_Panel/MeshInstance_316/______1"
        # self.PANEL_URL = "omniverse://localhost/Projects/ETRI/Env/Defection/Car_Clouser_Door.usd"
        
        self._target_prim = "/World/Car_Panel/Xform/Mesh_703/Mesh_703"
        self.PANEL_URL = "omniverse://localhost/Projects/ETRI/Env/Defection/NosePanel.usdc"
        
        self._texture_directory = current_dir + "/textures"
        return

    def add_defect_obj(
            self, 
            translate=(0.0, 0.0, 0.0),
            name="Car_Panel"
        ):
        prim_path = f"/World/{name}"
        add_object(
            usd_path=self.PANEL_URL,
            prim_path=prim_path,
            translate=translate,
            orient=(1.0, 0.0, 0.0, 0.0),
            scale=0.001,
        )
        add_semantic_data(prim_path, "car_panel")

    def create_defects(
            self,
            target_prim: str,
            texture_directory: str,
            abn_type: str = "scratches",
            semantic: str = "scratch_mesh",
            num_abn: int = 3,
            min_rotation: tuple = (-180, -180, -180),
            max_rotation: tuple = (180, 180, 180),
            min_scale: tuple = (0.1, 0.1, 0.1),
            max_scale: tuple = (0.1, 0.1, 0.1),
        ):

        # 지정된 texture_directory안에는 D.png, _N.png, _R.png로 끝나는 파일들이 있다.
        # 이들은 모두 스크레치를 표현하기 위한 texture들로, 각각 diffuse, normal, roughness texture이다.
        texture_folder_directory = texture_directory + "/" + abn_type

        diffuse_textures = get_textures(texture_folder_directory, "_D.png")
        normal_textures = get_textures(texture_folder_directory, "_N.png")
        roughness_textures = get_textures(texture_folder_directory, "_R.png")
        num_scratch_textures = len(diffuse_textures)

        # target_prim에 해당하는 prim을 가져온다.
        target_prim_rep = rep.get.prims(
            path_pattern=target_prim
        )
        
        i = 0

        cube_sem = [('class', f'{semantic}_{i}')]
        proj_sem = [('class', f'{semantic}_projectmat')]

        # 물체 위에 생성할 스크레치 개수 만큼 cube를 생성하고, projection_material을 추가한다.
        # projection_material이란 이미지 형식의 스크레치를 각진 물체 위에 사영(projection)하기 위한 Material이다.
        for i in range(num_abn):
            cube = rep.create.cube(
                visible=False,
                semantics=cube_sem, 
                position=0,
                scale=0.0001,
                rotation=(0, 0, 90)
            )
            with target_prim_rep:
                rep.create.projection_material(
                    cube,
                    proj_sem,
                    offset_scale=0.0001
                )

        # 랜덤한 스크레치를 위해 큐브 Prim을 랜덤화시킨다.
        # 이후, 랜덤 큐브 위로 이미지가 덮어씌워질 것이다.
        defects = rep.get.prims(semantics=cube_sem)
        plane = rep.get.prim_at_path(target_prim)
        
        with defects:
            rep.randomizer.scatter_2d(plane)
            rep.modify.pose(
                rotation=rep.distribution.uniform(
                    min_rotation, 
                    max_rotation
                ),
                scale=rep.distribution.uniform(
                    min_scale,
                    max_scale
                )
            )

        projections = rep.get.prims(semantics=[('class', f'{semantic}_projectmat')])
        
        # texture_folder_directory안에 다양한 스크레치 이미지가 있다.
        # 이들 중 랜덤한 이미지를 선택한다.
        random_number = random.choice(
            list(range(num_scratch_textures))
        )

        # 최종 작업. 랜덤한 스크레치 이미지를 랜덤 큐브 위에 덮어씌운다.
        with projections:
            rep.modify.projection_material(
                diffuse=diffuse_textures[random_number],
                normal=normal_textures[random_number],
                roughness=roughness_textures[random_number]
            )
        return


    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.add_defect_obj(
            translate=(0.0, 0.0, 0.3),
            name="Car_Panel"
        )
        return


    async def setup_post_load(self):

        self.create_defects(
            target_prim=self._target_prim,
            texture_directory=self._texture_directory,
            abn_type="scratches",
            semantic="scratch_mesh",
            num_abn=20,
            min_rotation=(-180, -180, -180),
            max_rotation=(180, 180, 180),
            min_scale=(0.1, 0.1, 0.1),
            max_scale=(0.1, 0.1, 0.1)
        )

        # self.create_defects(
        #     target_prim=self._target_prim,
        #     texture_directory=self._texture_directory,
        #     abn_type="stains",
        #     semantic="stain_mesh",
        #     num_abn=20,
        #     min_rotation=(-180, -180, -180),
        #     max_rotation=(180, 180, 180),
        #     min_scale=(0.1, 0.1, 0.1),
        #     max_scale=(0.1, 0.1, 0.1)
        # )

        # self.create_defects(
        #     target_prim=self._target_prim,
        #     texture_directory=self._texture_directory,
        #     abn_type="holes",
        #     semantic="hole_mesh",
        #     num_abn=20,
        #     min_rotation=(-180, -180, -180),
        #     max_rotation=(180, 180, 180),
        #     min_scale=(0.1, 0.1, 0.1),
        #     max_scale=(0.1, 0.1, 0.1)
        # )
        return

