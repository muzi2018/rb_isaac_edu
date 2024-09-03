from omni.isaac.etri_env import add_sphere_light, add_random_light, create_random_mdl_materaial
from omni.isaac.etri_env import add_object, add_random_object_visual, add_semantic_data
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.etri_env import add_physical_properties

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema

import numpy as np
import json
import omni

# TODO
# json을 받아서
# 배경 경로, 배경 위치, 배경 회전, 배경 크기
# 빛 추가
# 물체 경로, 물체 이름, 물체 위치, 물체 회전, 물체 크기,
# 물체 부피 타입, 물체 rigid body 줄건지, GPU Dynamics 켤건지,
# 물체 materail
# 를 받아서 물체를 추가하는 함수를 만들어야 한다.

# 일단 내가 환경 하나 만든다.
# 그 환경을 json으로 스폰 가능하도록 한다.

# replicator light

# 랜덤화 적용


# Function to read and parse the JSON file
def read_and_parse_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Parse JSON file
            return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

class SemiconductorRandomizerExample():

    def __init__(
            self,
            world,
            json_file_path=None,
        ) -> None:
        
        self._world = world
        self._stage = omni.usd.get_context().get_stage()
        self._json_file_path = json_file_path
        self._env_config = read_and_parse_json(self._json_file_path)

        self._base_path = "omniverse://localhost/Projects/ETRI/Env"

        self._bg_path = self._base_path + "/BackGrounds/Simple_Factory_Scene.usdz"
        self._shelf_path = self._base_path + "/RobotRack/Worn_Warehouse_Shelf.usdz"
        
        self._parse_json_config()

        return
    
    def _parse_json_config(self):
        self._is_background = self._env_config.get("is_background", False)
        self._bg_config = self._env_config.get("background", {})
        self._lights_config = self._env_config.get("lights", [])
        self._objects_config = self._env_config.get("objects", [])
        return


    def add_background(self):
        prim_path = f"/World/Background"
        scale = self._bg_config.get("background_scale", 1.0)
        trans_x, trans_y, trans_z = self._bg_config.get("background_trans", (0.0, 0.0, 0.0))
        q_w, q_x, q_y, q_z = self._bg_config.get("background_orient", (1.0, 0.0, 0.0, 0.0))

        add_object(
            usd_path=self._bg_config.get("background_url", self._bg_path),
            prim_path=prim_path,
            translate=(trans_x, trans_y, trans_z),
            orient=(q_w, q_x, q_y, q_z),
            scale=scale,
        )
        add_semantic_data(prim_path, "background")
        return
    
    # self.add_random_lights()
    # self.add_random_objects()

    def add_random_lights(self):
        
        for light_config in self._lights_config:
            print(f"light_config : {light_config}")

            add_random_light(
                min_trans=light_config["min_trans"], 
                max_trans=light_config["max_trans"], 
                min_rotation=light_config["min_rotation"], 
                max_rotation=light_config["max_rotation"],
                min_scale=light_config["min_scale"], 
                max_scale=light_config["max_scale"],
                light_type=light_config["light_type"],
                min_color=light_config["min_color"], 
                max_color=light_config["max_color"],
                min_intensity=light_config["min_intensity"], 
                max_intensity=light_config["max_intensity"],
                min_temperature=light_config["min_temperature"], 
                max_temperature=light_config["max_temperature"],
            )

        # add_random_light(
        #     min_trans=(-600, 0.0, 220), max_trans=(-600, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(-450, 0.0, 220), max_trans=(-450, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(-300, 0.0, 220), max_trans=(-300, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(-150, 0.0, 220), max_trans=(-150, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(-0, 0.0, 220), max_trans=(-0, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(150, 0.0, 220), max_trans=(150, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(300, 0.0, 220), max_trans=(300, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(450, 0.0, 220), max_trans=(450, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # add_random_light(
        #     min_trans=(600, 0.0, 220), max_trans=(600, 0.0, 220),
        #     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0),
        #     min_scale=(20.0, 20.0, 20.0), max_scale=(20.0, 20.0, 20.0),
        #     light_type="sphere",
        #     min_color=(1, 1, 1), max_color=(1, 1, 1),
        #     min_intensity=100000, max_intensity=100000,
        #     min_temperature=6500, max_temperature=6500,
        # )

        # # -600, 0.0, 220.0 / -450, 0.0, 220.0 / -300, 0.0, 220.0 / 
        # # -150, 0.0, 220.0 / 0.0, 0.0, 220.0 / 150, 0.0, 220.0 / 
        # # 300, 0.0, 220.0 / 450, 0.0, 220.0 / 600, 0.0, 220.0
        # # intensity: 100000.0
        # # radius: 10.0


    def add_random_objects(self):
        add_random_object_visual(
            min_trans=(-200, -200, 0.0), max_trans=(-200, -200, 0.0),
            min_rotation=(90, 0, 0), max_rotation=(90, 0, 0),
            min_scale=0.5, max_scale=0.5,
            usd_path=self._shelf_path, prim_path="/World/Shelf0", semantic_class="shelf",
        )

        add_random_object_visual(
            min_trans=(-200, 200, 0.0), max_trans=(-200, 200, 0.0),
            min_rotation=(90, 0, 0), max_rotation=(90, 0, 0),
            min_scale=0.5, max_scale=0.5,
            usd_path=self._shelf_path, prim_path="/World/Shelf1", semantic_class="shelf",
        )

        add_random_object_visual(
            min_trans=(200, -200, 0.0), max_trans=(200, -200, 0.0),
            min_rotation=(90, 0, 0), max_rotation=(90, 0, 0),
            min_scale=0.5, max_scale=0.5,
            usd_path=self._shelf_path, prim_path="/World/Shelf2", semantic_class="shelf",
        )

        add_random_object_visual(
            min_trans=(200, 200, 0.0), max_trans=(200, 200, 0.0),
            min_rotation=(90, 0, 0), max_rotation=(90, 0, 0),
            min_scale=0.5, max_scale=0.5,
            usd_path=self._shelf_path, prim_path="/World/Shelf3", semantic_class="shelf",
        )

    def add_objects(self):
        
        prim_path = f"/World/Shelf0"
        add_object(
            usd_path=self._shelf_path,
            prim_path=prim_path,
            translate=(-200.0, -200.0, 0.0),
            orient=(
                0.7071068, 0.7071068, 0, 0,
            ),
            scale=(0.5, 0.5, 0.5),
        )
        add_semantic_data(prim_path, "shelf")

        prim_path = f"/World/Shelf1"
        add_object(
            usd_path=self._shelf_path,
            prim_path=prim_path,
            translate=(-200.0, 200.0, 0.0),
            orient=(
                0.7071068, 0.7071068, 0, 0,
            ),
            scale=(0.5, 0.5, 0.5),
        )  
        add_semantic_data(prim_path, "shelf")

        prim_path = f"/World/Shelf2"
        add_object(
            usd_path=self._shelf_path,
            prim_path=prim_path,
            translate=(200.0, -200.0, 0.0),
            orient=(
                0.7071068, 0.7071068, 0, 0,
            ),
            scale=(0.5, 0.5, 0.5),
        )
        add_semantic_data(prim_path, "shelf")

        prim_path = f"/World/Shelf3"
        add_object(
            usd_path=self._shelf_path,
            prim_path=prim_path,
            translate=(200.0, 200.0, 0.0),
            orient=(
                0.7071068, 0.7071068, 0, 0,
            ),
            scale=(0.5, 0.5, 0.5),
        )
        add_semantic_data(prim_path, "shelf")

        # (-200 -200 0)
        # (-200 200 0)
        # (200 -200 0)
        # (200 200 0)

        return

    def add_lights(self):

        for i in range(9):
            add_sphere_light(
                self._stage,
                prim_path=f"/World/Light{i}",
                intensity=100000.0,
                translate=(-600 + 150 * i, 0.0, 220.0),
                scale=10.0,
            )
        
        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light0",
        #     intensity=100000.0,
        #     translate=(-600, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light1",
        #     intensity=100000.0,
        #     translate=(-450, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light2",
        #     intensity=100000.0,
        #     translate=(-300, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light3",
        #     intensity=100000.0,
        #     translate=(-150, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light4",
        #     intensity=100000.0,
        #     translate=(0.0, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light5",
        #     intensity=100000.0,
        #     translate=(150, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light6",
        #     intensity=100000.0,
        #     translate=(300, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light7",
        #     intensity=100000.0,
        #     translate=(450, 0.0, 220.0),
        #     scale=10.0,
        # )

        # add_sphere_light(
        #     self._stage,
        #     prim_path="/World/Light8",
        #     intensity=100000.0,
        #     translate=(600, 0.0, 220.0),
        #     scale=10.0,
        # )


        # -600, 0.0, 220.0 / -450, 0.0, 220.0 / -300, 0.0, 220.0 / -150, 0.0, 220.0 / 0.0, 0.0, 220.0 / 150, 0.0, 220.0 / 300, 0.0, 220.0 / 450, 0.0, 220.0 / 600, 0.0, 220.0
        # intensity: 100000.0
        # radius: 10.0

    def setup_scene(self):

        if self._is_background:
            self.add_background()
        
        self.add_random_lights()
        # # self.add_lights()
        
        # # self.add_objects()
        # self.add_random_objects()

        # self.add_materials()
        return

    def add_background_properties(self):
        prim_path = f"/World/Background"
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=False,
            collision_approximation="meshSimplification"
        )
        return
    
    def add_objects_physical_properties(self):
        prim_paths = [
            f"/World/Shelf0",
            f"/World/Shelf1",
            f"/World/Shelf2",
            f"/World/Shelf3",
        ]
        
        for prim in prim_paths:
            add_physical_properties(
                prim_path=prim,
                rigid_body=True,
                collision_approximation="none"
            )
        return
    
    def add_material_properties(self):
        prim_path = self._stage.GetPrimAtPath(f"/World/Shelf0")
        UsdShade.MaterialBindingAPI(prim_path).Bind(
            self._metal_material, 
            UsdShade.Tokens.strongerThanDescendants
        )
        return

    async def setup_post_load(self):
        self.add_background_properties()
        # self.add_objects_physical_properties()
        # self.add_material_properties()
        
        await self._world.reset_async()
        
        return
