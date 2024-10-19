from .multi_env_utils import (
    add_random_light,
    turn_off_dg_light,
    add_bg,
    add_camera,
    add_semantic_data,
    add_random_object_visual,
    add_random_object_physics,
    add_random_object_port_physics,
    add_cloth_particle_physics,
    add_gripper_friction,
    create_defects,
    create_random_omnipbr_material,
    create_random_mdl_materaial,
    create_random_omniglass_material,
    attach_material_to_prim,
)
from .robo_config import ROBO_USD_DICT

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema

from omni.isaac.robo_data_collector import RoboDataCollector
from omni.isaac.custom_franka import Franka3F, Franka2F
from omni.isaac.franka import Franka

from omni.isaac.ur5 import UR52F, UR53F
from omni.isaac.custom_ur10 import UR102F, UR103F
from omni.isaac.zeus import Zeus2F, Zeus3F

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep

import numpy as np 
import logging
import omni
import json
import carb
import time


ENV_BASE_PATH = "omniverse://localhost/Projects/ETRI/Env"


# Function to read and parse the JSON file
def read_and_parse_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Parse JSON file
            return data
    except FileNotFoundError:
        logging.error(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the file {file_path}.")
    except Exception as e:
        logging.error(f"""
            An error occurred During JSON parsing : {e} \n
            Please Check whether useless comma is exists in the JSON file 
        """)
    return


class MultiEnvRandomizer():

    def __init__(
            self,
            world,
            json_file_path=None,
            verbose=True,
        ) -> None:

        self._world = world
        self._verbose = verbose
        self._json_file_path = json_file_path

        import datetime
        now = datetime.datetime.now()
        self._now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

        self._scene = PhysicsContext()
        self._stage = omni.usd.get_context().get_stage()
        self._env_config = read_and_parse_json(self._json_file_path)

        self._save_count = 0
        self._capture_count = 0
        self._multi_robot_config = None
        
        self._robots = []
        self._rep_cams = [] # replicator cameras
        self._abn_objects = [] # abnormal objects
        self._controllers = [] # robot controllers
        self._picking_events = []
        self._robo_data_collectors = []
        self._interactive_object_config = []
        
        self._pbr_material_objects = []
        self._mdl_material_objects = []
        self._omniglass_material_objects = []
        self._parse_json_config()
        return


    def _parse_json_config(self):
        try:
            self._is_background = self._env_config.get("is_background", False)
            self._enable_gpu_dynamics = self._env_config.get("enable_gpu_dynamics", False)
            self._invert_collision_group_filter = self._env_config.get("invert_collision_group_filter", False)
            self._collect_robo_data = self._env_config.get("collect_robo_data", False)
            self._collect_vision_data = self._env_config.get("collect_vision_data", False)
            self._env_offsets = self._env_config.get("env_offsets", [])
            self._bg_config = self._env_config.get("background", {})
            self._lights_config = self._env_config.get("lights", [])
            self._bg_objects_config = self._env_config.get("background_objects", [])
            self._interactive_objects_config = self._env_config.get("interactive_objects", [])
            
            # multi robot special case
            if "robots" in self._env_config:
                self._multi_robot_config = self._env_config.get("robots", {})
            
            self._robot_config = self._env_config.get("robot", {})
            self._cameras_config = self._env_config.get("cameras", {})
            self._task_config = self._env_config.get("task", {})
            self._vision_data_config = self._env_config.get("vision_data", {})
            self._robo_data_config = self._env_config.get("robo_data", {})
        except Exception as e:
            logging.error(f"""
                Failed to parse json config : {e}
                Please check the json file path: {self._json_file_path}
                And check all parts exist in the json file or check the comma in
            """)
            return


    def add_xform(self, index, env_offset):
        xform = prims_utils.define_prim(f"/World/Env_{index}", prim_type="Xform")
        xformable = UsdGeom.Xformable(xform)
        xformable.AddTranslateOp().Set(value=tuple(env_offset))


    def add_background(self, index, env_offset):

        prim_path = f"/World/Env_{index}/Background"

        usd_path = self._bg_config.get("background_url", "")
        trans_x, trans_y, trans_z = self._bg_config.get("background_trans", (0.0, 0.0, 0.0))
        q_w, q_x, q_y, q_z = self._bg_config.get("background_orient", (1.0, 0.0, 0.0, 0.0))
        scale = self._bg_config.get("background_scale", 1.0)
        collision_approximation = self._bg_config.get("collision_approximation", "meshSimplification")

        add_bg(
            usd_path=usd_path,
            prim_path=prim_path,
            translate=(trans_x, trans_y, trans_z),
            orient=(q_w, q_x, q_y, q_z),
            scale=scale,
            collision_approximation=collision_approximation
        )
    

    # async def add_random_lights(self, env_offset):
    def add_random_lights(self, env_offset):

        for light_config in self._lights_config:

            min_trans = [x + y for x, y in zip(light_config["min_trans"], env_offset)]
            max_trans = [x + y for x, y in zip(light_config["max_trans"], env_offset)]

            add_random_light(
                min_trans=min_trans,
                max_trans=max_trans,
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
        # await self._world.reset_async()


    def port_physics_helper(
            self, 
            prim_path, 
            random_trans,
            random_rot,
            random_scale,
            object_config
        ):

        if self._verbose:
            logging.warning(f"[Port Physics Helper] Adding Port Physics to {prim_path}")

        physics_config = object_config["port_physics"]

        if physics_config["mass_modify"] == True:
            min_mass_value = physics_config["min_mass_value"]
            max_mass_value = physics_config["max_mass_value"]
        else:
            min_mass_value = -1.0
            max_mass_value = -1.0

        mass_value = np.random.uniform(min_mass_value, max_mass_value)

        if physics_config["friction_modify"] == True:
            min_static_friction = physics_config["min_static_friction"]
            max_static_friction = physics_config["max_static_friction"]
            min_dynamic_friction = physics_config["min_dynamic_friction"]
            max_dynamic_friction = physics_config["max_dynamic_friction"]

            static_friction = np.random.uniform(min_static_friction, max_static_friction)
            dynamic_friction = np.random.uniform(min_dynamic_friction, max_dynamic_friction)
        else:
            static_friction = None
            dynamic_friction = None

        geom = add_random_object_port_physics(
            world=self._world,
            prim_path=prim_path,
            ini_pos=random_trans,
            orientation=random_rot,
            scale=random_scale,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            kinematic_enabled=physics_config["is_kinematic"],
            collision_type=physics_config["collision_type"],
            mass_value=mass_value,
        )
        if geom is not None: 
            self._world.scene.add(geom)

        return geom


    def physics_helper(self, prim_path, physics_config):

        if self._verbose:
            logging.warning(f"[Physics Helper] Adding Physics to {prim_path}")

        # Special Object cases
        if ("is_cloth_particle" in physics_config) and (physics_config["is_cloth_particle"] == True):
            add_cloth_particle_physics(
                stage=self._stage,
                prim_path=prim_path,
                physics_config=physics_config,
            )
        else:
            pass

        if physics_config["is_collision"] == True:
            collision_type = physics_config["collision_type"]
        else:
            collision_type = "none"

        if physics_config["mass_modify"] == True:
            min_mass_value = physics_config["min_mass_value"]
            max_mass_value = physics_config["max_mass_value"]
        else:
            min_mass_value = -1.0
            max_mass_value = -1.0

        if physics_config["friction_modify"] == True:
            min_static_friction = physics_config["min_static_friction"]
            max_static_friction = physics_config["max_static_friction"]
            min_dynamic_friction = physics_config["min_dynamic_friction"]
            max_dynamic_friction = physics_config["max_dynamic_friction"]
        else:
            min_static_friction = -1.0
            max_static_friction = -1.0
            min_dynamic_friction = -1.0
            max_dynamic_friction = -1.0

        geom, rigid = add_random_object_physics(
            prim_path=prim_path,
            is_rigid_body=physics_config["is_rigid_body"],
            is_collision=physics_config["is_collision"],
            is_kinematic=physics_config["is_kinematic"],
            collision_type=collision_type,
            mass_modify=physics_config["mass_modify"],
            min_mass_value=min_mass_value,
            max_mass_value=max_mass_value,
            friction_modify=physics_config["friction_modify"],
            min_static_friction=min_static_friction,
            max_static_friction=max_static_friction,
            min_dynamic_friction=min_dynamic_friction,
            max_dynamic_friction=max_dynamic_friction,
        )
        if geom is not None: 
            self._world.scene.add(geom)

        return geom, rigid


    def material_helper(self, prim_path, material_config):

        if material_config["material_type"] == "omnipbr":
            create_random_omnipbr_material(
                prim_path=None,
                min_diffuse=material_config["min_diffuse"],
                max_diffuse=material_config["max_diffuse"],
                min_roughness=material_config["min_roughness"],
                max_roughness=material_config["max_roughness"],
                min_metallic=material_config["min_metallic"],
                max_metallic=material_config["max_metallic"],
                min_specular=material_config["min_specular"],
                max_specular=material_config["max_specular"],
                min_emissive_color=material_config["min_emissive_color"],
                max_emissive_color=material_config["max_emissive_color"],
                min_emissive_intensity=material_config["min_emissive_intensity"],
                max_emissive_intensity=material_config["max_emissive_intensity"],
            )
            self._pbr_material_objects.append(prim_path)
        elif material_config["material_type"] == "mdl":
            create_random_mdl_materaial(
                prim_path=prim_path,
                category=material_config["category"],
                mtl_name=material_config["mtl_name"],
            )
            self._mdl_material_objects.append(prim_path)
        elif material_config["material_type"] == "omniglass":
            create_random_omniglass_material(
                stage=self._stage,
                prim_path=prim_path,
                min_color=material_config["min_color"],
                max_color=material_config["max_color"],
                min_depth=material_config["min_depth"],
                max_depth=material_config["max_depth"],
                min_ior=material_config["min_ior"],
                max_ior=material_config["max_ior"],
            )
            self._omniglass_material_objects.append(prim_path)


    def object_abnormal_helper(self, prim_path, object_config):
        try:
            abn_obj_dict = {
                prim_path: object_config
            }
            self._abn_objects.append(abn_obj_dict)
        except Exception as e:
            logging.error(f"Error occured during Adding Object Abnormal : {e}")
            logging.error(f"Please complete all parts in json!")
            return


    def add_random_bg_objects(self, index, env_offset):
        for object_config in self._bg_objects_config:
            object_name = object_config["name"]
            prim_path = f"/World/Env_{index}/{object_name}"

            add_random_object_visual(
                usd_path=object_config["object_url"],
                prim_path=prim_path,
                semantic_class=object_config["semantic_name"],
                min_trans=object_config["min_trans"],
                max_trans=object_config["max_trans"],
                min_rotation=object_config["min_rotation"], 
                max_rotation=object_config["max_rotation"],
                min_scale=object_config["min_scale"], 
                max_scale=object_config["max_scale"],
            )

            if "material" in object_config:
                self.material_helper(
                    prim_path, 
                    object_config["material"]
                )


    def add_random_interactive_objects(self, index):
        for object_config in self._interactive_objects_config:
            object_name = object_config["name"]
            prim_path = f"/World/Env_{index}/{object_name}"

            try:
                random_trans, random_rot, random_scale = add_random_object_visual(
                    usd_path=object_config["object_url"],
                    prim_path=prim_path,
                    semantic_class=object_config["semantic_name"],
                    min_trans=object_config["min_trans"],
                    max_trans=object_config["max_trans"],
                    min_rotation=object_config["min_rotation"], 
                    max_rotation=object_config["max_rotation"],
                    min_scale=object_config["min_scale"], 
                    max_scale=object_config["max_scale"],
                )

                # particle, cloth, deformable과 같은 spceical object는 
                # 물성이 시뮬레이션 구성 도중 만들어지기 때문에 pose는 forward에서 직접 조회해야 한다.
                if "cloth" in object_name.lower() or "deformable" in object_name.lower():
                    prim = self._stage.GetPrimAtPath(prim_path)
                    obj_pose = omni.usd.get_world_transform_matrix(prim)

                if self._task_config["task_type"] == "insertion" and object_name in self._task_config["picking_object_name"]:
                    # picking_object는 GeometryPrim은 무조건 적용해준다. 
                    if not ("physics" in object_config):
                        geom_name = prim_path.split("/")[-2] + "_" + prim_path.split("/")[-1]
                        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=geom_name))

                        if self._verbose:
                            logging.warning("[Insertion Task] Adding Picking Object to the scene")

                    if "cable" in object_name.lower():
                        fc_prim_path = prim_path + "/Front_Connector"
                        fc_prim = self._stage.GetPrimAtPath(fc_prim_path)
                        fc_pose = omni.usd.get_world_transform_matrix(fc_prim)

                if self._task_config["task_type"] == "peg_in_hole" and "port_physics" in object_config:
                    pass

                if self._task_config["task_type"] == "screw" and ("nut" in object_name.lower() or "bolt" in object_name.lower()):
                    # screw 시 bolt, nut는 GeometryPrim은 무조건 적용해준다. 
                    geom_name = prim_path.split("/")[-2] + "_" + prim_path.split("/")[-1]

                    if self._verbose:
                        logging.warning(f"[Screw Task] Adding {geom_name} to the scene")

                    self._world.scene.add(GeometryPrim(prim_path=prim_path, name=geom_name))

                if "physics" in object_config:
                    geom, rigid = self.physics_helper(
                        prim_path,
                        physics_config=object_config["physics"],
                    )
                elif "port_physics" in object_config:
                    geom = self.port_physics_helper(
                        prim_path,
                        random_trans, 
                        random_rot, 
                        random_scale,
                        object_config=object_config,
                    )

                if "material" in object_config:
                    self.material_helper(
                        prim_path, 
                        material_config=object_config["material"]
                    )

                if "object_abnormal" in object_config:
                    self.object_abnormal_helper(
                        prim_path,
                        object_config=object_config["object_abnormal"]
                    )

            except Exception as e:
                logging.error(f"Error occured during Adding Interactive Objects : {e}")
                logging.error(f"Please complete all parts in json!")
                return
    
    def add_multi_robots(self, index, env_offset):
        if self._multi_robot_config is None:
            logging.error(f"Multi Robot Config is None")
            return

        for robot_config in self._multi_robot_config:
            self._robot_config = robot_config
            self.add_robot(index, env_offset, robot_name=robot_config["robot_name"])
        return

    def add_robot(self, index, env_offset, robot_name=None):
        
        try:
            self._robot_type = self._robot_config["robot_type"]
            self._gripper_type = self._robot_config["gripper_type"]
            if "sensor_type" in self._robot_config:
                self._sensor_type = self._robot_config["sensor_type"]
            else:
                self._sensor_type = "none"
            if "gripper_mode" in self._robot_config:
                self._gripper_mode = self._robot_config["gripper_mode"]
            else:
                self._gripper_mode = "basic"
            min_trans = self._robot_config["min_trans"]
            max_trans = self._robot_config["max_trans"]
            min_rotation = self._robot_config["min_rotation"]
            max_rotation = self._robot_config["max_rotation"]
        except Exception as e:
            logging.error(f"Error occured during Adding Robot : {e}")
            logging.error(f"Please complete all parts in json!")
            return

        # random translation btw min_trans - max_trans
        self._robot_position = np.random.uniform(min_trans, max_trans)
        self._robot_rotation_euler = np.random.uniform(min_rotation, max_rotation)
        # degree to radian
        self._robot_rotation_euler = np.deg2rad(self._robot_rotation_euler)
        self._robot_rotation_quat = euler_angles_to_quat(self._robot_rotation_euler)

        self._robot_position = [x + y for x, y in zip(self._robot_position, env_offset)]
    
        usd_path = None
        if robot_name is None:
            prim_path = f"/World/Env_{index}/robot"
            robot_name = f"robot_{index}"
        else:
            prim_path = f"/World/Env_{index}/{robot_name}"
            robot_name = f"{robot_name}_{index}"

        robot_class_mapping = {
            "franka": Franka if self._gripper_type == "normal" else Franka2F if self._gripper_type == "2f85" else Franka3F,
            "ur5": UR52F if self._gripper_type == "2f85" else UR53F,
            "ur10": UR102F if self._gripper_type == "2f85" else UR103F,
            "zeus": Zeus2F if self._gripper_type == "2f85" else Zeus3F
        }

        robot_class = robot_class_mapping[self._robot_type]
        usd_path = ROBO_USD_DICT[self._robot_type][self._gripper_type][self._sensor_type]

        if self._robot_type == "franka" and self._gripper_type == "normal":
            robot = self._world.scene.add(
                robot_class(
                    usd_path=usd_path,
                    prim_path=prim_path,
                    name=robot_name,
                    position=self._robot_position,
                    orientation=self._robot_rotation_quat,
                )
            )
        else:
            if self._gripper_mode == "custom":
                usd_path = ROBO_USD_DICT[self._robot_type][self._gripper_mode]
                custom_open_positions = self._robot_config["custom_open_positions"]
                custom_closed_positions = self._robot_config["custom_closed_positions"]
            else:
                custom_open_positions = None
                custom_closed_positions = None
            robot = self._world.scene.add(
                robot_class(
                    usd_path=usd_path,
                    prim_path=prim_path,
                    name=robot_name,
                    position=self._robot_position,
                    orientation=self._robot_rotation_quat,
                    attach_gripper=True,
                    gripper_mode=self._gripper_mode,
                    custom_open_positions=custom_open_positions,
                    custom_closed_positions=custom_closed_positions
                )
            )
        add_semantic_data(prim_path, "robot")

        time.sleep(1)
        self._robots.append(robot)
        return

    
    def add_bg_camera(self, i, env_offset):
        for camera_config in self._cameras_config:
            prim_path = f"/World/Env_{i}/" + camera_config["name"]
            global_position = [x + y for x, y in zip(camera_config["position"], env_offset)]
            add_camera(
                prim_path=prim_path,
                position=global_position,
                orientation=camera_config["orientation"],
                focal_length=camera_config["focal_length"]
            )
        return

    def setup_dataset(self, index):
        return


    def screw_setup_simulation(self):
        return


    def add_bg_repliactor(self, i):

        if not ("background_cameras" in self._vision_data_config):
            logging.warning(f"Not Background Camera Specified Skip this Step")
            return

        bg_camera_configs = self._vision_data_config["background_cameras"]
        
        for bg_camera_config in bg_camera_configs:
            prim_path = f"/World/Env_{i}/" + bg_camera_config["prim_path"]

            resolution = tuple(bg_camera_config["resolution"])
            name = bg_camera_config["name"]

            rp_bg = rep.create.render_product(
                prim_path,
                resolution,
                name=name,
                force_new=False
            )
            self._rep_cams.append(rp_bg)

        return


    def add_manipulator_replicator(self, i):

        if not ("robot_camera" in self._vision_data_config):
            logging.warning(f"Not Robot Camera Specified Skip this Step")
            return

        try:
            robot_camera_config = self._vision_data_config["robot_camera"]
            prim_path = f"/World/Env_{i}/robot/" + robot_camera_config["prim_path"]
            resolution = tuple(robot_camera_config["resolution"])
            name = robot_camera_config["name"]
        except Exception as e:
            logging.error(f"Error occured during Adding Env {i} Manipulator Vision Data : {e}")
            logging.error(f"Please complete all parts in json!")
            return

        self._rp_ee = rep.create.render_product(
            prim_path,
            resolution,
            name=name,
            force_new=False
        )
        self._rep_cams.append(self._rp_ee)


    def setup_replicator(self, i):

        try:
            capture_hz = self._vision_data_config["capture_hz"]
            physics_dt = self._scene.get_physics_dt()
            frame_rate = (1/capture_hz) / physics_dt
        except Exception as e:
            logging.error(f"Error occured during Adding Env {i} Replicator Setup : {e}")
            logging.error(f"Please complete all parts in json!")
            return

        if len(self._rep_cams) < 1:
            logging.warning(f"No Replicator Camera Found")
            return

        folder_name = f"/{self._task_type}_{self._robot_type}_{self._now_str}/Env_{i}"
        out_dir = self._vision_data_config["out_dir"] + folder_name

        if self._verbose:
            logging.warning(f"[Vision Data] Writing data to: {out_dir}")
        
        data_type = self._vision_data_config["data_type"]

        try:
            rgb = data_type["rgb"]
            bounding_box_2d_tight = data_type["bounding_box_2d_tight"]
            bounding_box_2d_loose = data_type["bounding_box_2d_loose"]
            semantic_segmentation = data_type["semantic_segmentation"]
            instance_id_segmentation = data_type["instance_id_segmentation"]
            instance_segmentation = data_type["instance_segmentation"]
            bounding_box_3d = data_type["bounding_box_3d"]
            distance_to_camera = data_type["distance_to_camera"]
            pointcloud = data_type["pointcloud"]
            if "colorize_semantic_segmentation" in data_type:
                colorize_semantic_segmentation = data_type["colorize_semantic_segmentation"]
                colorize_instance_id_segmentation = data_type["colorize_instance_id_segmentation"]
                colorize_instance_segmentation = data_type["colorize_instance_segmentation"]
        except Exception as e:
            logging.error(f"Error occured during Adding Env {i} Vision Data : {e}")
            logging.error(f"Please complete all parts in json!")

        self._writer = rep.WriterRegistry.get("CustomBasicWriter")
        self._writer.initialize(
            output_dir=out_dir,
            rgb=rgb,
            bounding_box_2d_tight=bounding_box_2d_tight,
            bounding_box_2d_loose=bounding_box_2d_loose,
            semantic_segmentation=semantic_segmentation,
            instance_id_segmentation=instance_id_segmentation,
            instance_segmentation=instance_segmentation,
            bounding_box_3d=bounding_box_3d,
            distance_to_camera=distance_to_camera,
            pointcloud=pointcloud,
            frame_rate=frame_rate
        )

        self._render_products = [
            *self._rep_cams
        ]
        self._writer.attach(self._render_products)


    def setup_scene(self):
        isregistry = carb.settings.acquire_settings_interface()        
        isregistry.set_bool("/rtx/directLighting/sampledLighting/autoEnable", False)
        
        # if self._task_config["task_type"] == "screw":
        #     self.screw_setup_simulation()

        if self._enable_gpu_dynamics:
            self._scene.set_solver_type("TGS")
            self._scene.set_broadphase_type("GPU")
            self._scene.enable_gpu_dynamics(flag=True)

        if self._invert_collision_group_filter:
            self._scene.set_invert_collision_group_filter(True)

        if self._is_background is False:
            # self._world.scene.add_ground_plane()
            self._world.scene.add_default_ground_plane()
            turn_off_dg_light()

        UsdGeom.Scope.Define(omni.usd.get_context().get_stage(), "/Looks/MDLMaterials")
        UsdGeom.Scope.Define(omni.usd.get_context().get_stage(), "/Looks/OmniGlassMaterials")
        self._world.scene.add(XFormPrim(prim_path="/World/collisionGroups", name="collision_groups_xform"))

        for i, env_offset in enumerate(self._env_offsets):

            if self._verbose:
                logging.warning(f"[MultiEnvRandomizer] Loading Env_{i}... env_offset: {env_offset}")

            self.add_xform(i, env_offset)

            if self._is_background:
                self.add_background(i, env_offset)
                self.add_random_bg_objects(i, env_offset)

            self.add_random_interactive_objects(i)

            if self._multi_robot_config is None:
                self.add_robot(i, env_offset)
            else:
                self.add_multi_robots(i, env_offset)

            self.add_bg_camera(i, env_offset)
        return


    def add_task(self, index, env_offset):

        try:
            self._task_type = self._task_config["task_type"]
        except Exception as e:
            logging.error(f"Error occured during Adding Task : {e}")
            logging.error(f"Please complete all parts in json!")
            return
        return


    def add_multi_robot_task(self, index, env_offset):

        if self._multi_robot_config is None:
            logging.error(f"Multi Robot Config is None")
            return

        try:
            self._task_type = self._task_config["task_type"]
        except Exception as e:
            logging.error(f"Error occured during Adding Task : {e}")
            logging.error(f"Please complete all parts in json!")
            return
        return


    def add_pbr_material_properties(self):
        for i, prim_path in enumerate(self._pbr_material_objects):
            if i == 0:
                material_name = f"/Replicator/Looks/OmniPBR"
            else:
                material_name = f"/Replicator/Looks/OmniPBR_{i:02d}"

            if self._verbose:
                logging.warning(f"[PBR Material] material_name: {material_name} / prim_path: {prim_path}")

            attach_material_to_prim(prim_path, material_name)


    def add_mdl_material_properties(self):
        for i, prim_path in enumerate(self._mdl_material_objects):
            material_name = "/Looks/MDLMaterials/" + prim_path.replace("/", "_").lower()

            if self._verbose:
                logging.warning(f"[MDL Material] material_name: {material_name} / prim_path: {prim_path}")

            attach_material_to_prim(prim_path, material_name)


    def add_omniglass_material_properties(self):
        for i, prim_path in enumerate(self._omniglass_material_objects):
            material_name = "/Looks/OmniGlassMaterials/" + prim_path.replace("/", "_").lower()

            if self._verbose:
                logging.warning(f"[OmniGlass Material] material_name: {material_name} / prim_path: {prim_path}")

            attach_material_to_prim(prim_path, material_name)


    async def add_abn_material_properties(self):
        for i, abn_obj_dict in enumerate(self._abn_objects):
            for prim_path, object_config in abn_obj_dict.items():

                target_prim = prim_path + "/" + object_config["mesh_path"]
                print(f"[add_abn_material_properties] Target Prim : {target_prim}")
                
                min_rotation = tuple(object_config["min_rotation"])
                max_rotation = tuple(object_config["max_rotation"])
                min_scale = tuple(object_config["min_scale"])
                max_scale = tuple(object_config["max_scale"])
                
                create_defects(
                    index=i,
                    target_prim=target_prim,
                    texture_directory=object_config["texture_directory"],
                    abn_type=object_config["type"],
                    semantic=object_config["semantic"],
                    num_abn=object_config["number"],
                    min_rotation=min_rotation,
                    max_rotation=max_rotation,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )

        await self._world.reset_async()
        return

    
    def setup_multi_robots(self, index):

        if self._multi_robot_config is None:
            logging.error(f"Multi Robot Config is None")
            return
        
        multi_robo_len = len(self._multi_robot_config)
        for i, robot_config in enumerate(self._multi_robot_config):
            robo_index = multi_robo_len * index + i
            self._robot_config = robot_config
            self.setup_robots(
                index, 
                robo_index=robo_index,
                robot_name=robot_config["robot_name"]
            )
        return


    def setup_robots(self, env_index, robo_index=None, robot_name=None):

        try:
            add_gripper_friction(
                index=env_index,
                world=self._world,
                robot_config=self._robot_config,
                robot_name=robot_name
            )

            if robo_index is None:
                robo_index = env_index
            else:
                robo_index = robo_index

            if "initial_joint_positions" in self._robot_config:
                if self._robot_config["robot_type"] == "franka":
                    self._robots[robo_index].set_joint_positions(
                        np.array(self._robot_config["initial_joint_positions"]),
                        joint_indices=np.array([0, 1, 2, 3, 4, 5, 6])
                    )
                else:
                    self._robots[robo_index].set_joint_positions(
                        np.array(self._robot_config["initial_joint_positions"]),
                        joint_indices=np.array([0, 1, 2, 3, 4, 5])
                    )
            self._robots[robo_index].gripper.open()

            if self._verbose:
                logging.warning(f"Robot {robo_index} is set up")
        except Exception as e:
            logging.error(f"Error occured during Setting Up Robot for Env {env_index} Robot : {e}")
            logging.error(f"Please complete all parts in json!")
            return
        return


    def setup_robo_dataset(self, i):
        self._save_count = 0
        self._capture_count = 0

        try:
            if i == 0:
                robo_data_capture_hz = self._robo_data_config["capture_hz"]
                physics_dt = self._scene.get_physics_dt()
                self._robo_data_frame_rate = int((1/robo_data_capture_hz) / physics_dt)
                logging.warning(f"Robo Data Frame Rate : {self._robo_data_frame_rate}")

            folder_name = f"/{self._task_type}_{self._robot_type}_{self._now_str}/Env_{i}/"
            file_path = self._robo_data_config["out_dir"] + folder_name + "robo_data.hdf5"
        except Exception as e:
            logging.error(f"Error occured during Adding Env {i} Robo Dataset : {e}")
            logging.error(f"Please complete all parts in json!")
            return

        robo_data_collector = RoboDataCollector(
            file_name=file_path
        )
        robot = self._robots[i]
        joint_names = robot._articulation_view.joint_names
        robo_data_collector.create_dataset(
            "joint_names",
            {
                f"{self._robot_type}_{self._gripper_type}": joint_names,
            }
        )

        self._robo_data_collectors.append(robo_data_collector)
        return


    async def setup_post_load(self):
        # add materials 
        self.add_pbr_material_properties()
        self.add_mdl_material_properties()
        self.add_omniglass_material_properties()
        await self.add_abn_material_properties()

        for i, env_offset in enumerate(self._env_offsets):
            # await self.add_random_lights(env_offset)
            self.add_random_lights(env_offset)

            if self._multi_robot_config is None:
                self.setup_robots(i)
                self.add_task(i, env_offset)
            else:
                self.setup_multi_robots(i)
                self.add_multi_robot_task(i, env_offset)

            if self._collect_vision_data:
                self._rep_cams = []
                self.add_bg_repliactor(i)
                self.add_manipulator_replicator(i)
                # TODO: 이거는 for logic 밖으로 빼기 
                self.setup_replicator(i)

            if self._collect_robo_data:
                self.setup_robo_dataset(i)
        
        # This halts initial_joint_positions
        # await self._world.reset_async()
        return


    def forward(self):

        if self._collect_robo_data:
            for i, (robot, robo_data_collector) in enumerate(zip(self._robots, self._robo_data_collectors)):
                if self._save_count % self._robo_data_frame_rate == 0:
                    frame_id = int(self._save_count / self._robo_data_frame_rate)

                    if self._verbose:
                        logging.warning(f"save_count : {self._save_count} / frame_id:  {frame_id} th frame")
                    
                    robo_data_collector.create_single_pp_data(
                        name=f"step_{frame_id:04}",
                        robot=robot,
                        controller=self._controllers[i],
                    )

        self._save_count += 1
        return


    def setup_pre_reset(self):
        if self._collect_robo_data:
            for robo_data_collector in self._robo_data_collectors:
                robo_data_collector.setup_pre_reset()
        self._save_count = 0
        self._capture_count = 0
        return


    def world_cleanup(self):
        if self._collect_robo_data:
            for i, robo_data_collector in enumerate(self._robo_data_collectors):
                robo_data_collector.save_data(i)
        self._save_count = 0
        self._capture_count = 0
        return
