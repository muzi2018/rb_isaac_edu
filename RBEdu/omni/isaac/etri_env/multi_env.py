from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from pxr import Usd, UsdGeom
import omni.graph.core as og

from os.path import expanduser

import numpy as np
import omni.usd
import omni
import carb

from omni.isaac.robo_data_collector import RoboDataCollector

class MultiEnv(object):
    def __init__(
            self,
            world=None,
            env_offsets: list = [0.0, 0.0, 0.0],
            add_background: bool = False,
            bg_path: str = None,
            add_replicator: bool = True,
            verbose: bool = False,
            robot_type: str = "2f85", # 2f85, 3f
            add_ros2: bool = False,
            collect_robo_data: bool = False,
        ) -> None:

        self._world = world
        self._env_offsets = env_offsets
        self._num_lens = len(self._env_offsets)

        self._index = 0
        self._xyz_offset = np.array([0.0, 0.0, 0.0])

        self._isaac_assets_path = get_assets_root_path()   # omniverse://localhost/NVIDIA/Assets/Isaac/4.0
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        self._robot_position = np.array([0.0, 0.0, 0.0])
        self._robot_orientation = np.array([1, 0, 0, 0])

        self._bg_scale = (1, 1, 1)
        self._bg_translation = (0, 0, 0)
        self._bg_orientation = (1, 0, 0, 0)

        self._add_bg = add_background
        self._add_rep = add_replicator
        self._verbose = verbose
        if self._verbose:
            print(f"[MultiRoboticRack] Num Lens: {self._num_lens}")
        
        self._save_count = 0
        self._robot_type = robot_type
        self._add_ros2 = add_ros2
        self._collect_robo_data = collect_robo_data

        if bg_path is not None:
            self.BG_PATH = self._server_root + "/Projects/ETRI/Env/" + bg_path
        else:
            self.BG_PATH = ""

        import datetime
        now = datetime.datetime.now()
        self._now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

        self._robots = []
        self._controllers = []
        self._motion_dict = {}
        self._robo_data_collectors = []
        return

    def set_motion_dict(self, motion_dict):
        self._motion_dict = motion_dict
        return

    def set_world(self, world):
        self._world = world
        return

    def set_env_offsets(self, env_offsets):
        self._env_offsets = env_offsets
        return

    def set_bg_pose(
            self, 
            bg_translation,
            bg_orientation,
            bg_scale,
        ):
        self._bg_translation = bg_translation
        self._bg_orientation = bg_orientation
        self._bg_scale = bg_scale
        return

    def add_bg(self):
        return

    def turn_off_dg_light(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
        attribute = prim.GetAttribute("intensity")
        attribute.Set(0)

    def add_camera(self):
        return

    def add_bg_repliactor(self):
        return

    def add_manipulator_replicator(self):
        return

    def add_og(self):
        return

    def add_xform(self):
        xform = prims_utils.define_prim(f"/World/Env_{self._index}", prim_type="Xform")
        xformable = UsdGeom.Xformable(xform)
        xformable.AddTranslateOp().Set(value=tuple(self._xyz_offset))
        return

    def add_light(self):
        return

    def add_robot(self):
        return

    def set_replicator_write_path(self, file_name):
        self._out_dir = str(expanduser("~") + "/Documents/ETRI_DATA/" + file_name)
        return

    def add_objects(self):
        return

    def setup_replicator(self):
        self._writer = rep.WriterRegistry.get("CustomBasicWriter")
        self._writer.initialize(
            output_dir=self._out_dir, 
            rgb=True,
            bounding_box_2d_tight=True,
            bounding_box_2d_loose=True,
            semantic_segmentation=True,
            instance_id_segmentation=True,
            instance_segmentation=True,
            bounding_box_3d=True,
            distance_to_camera=True,
            pointcloud=True,
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
        )
        return

    def setup_dataset(self):
        self._robo_data_collector = RoboDataCollector(file_name=f"robot{self._index}_data_" + self._now_str)
        self._save_count = 0

        self._robo_data_collectors.append(self._robo_data_collector)
        return

    def setup_scene(self):
        self._stage = omni.usd.get_context().get_stage()

        for i in range(len(self._env_offsets)):
            self._index = i
            self._xyz_offset = np.array(self._env_offsets[self._index])

            self.add_xform()
            self.add_camera()
            self.add_light()
            self.add_robot()
            self.add_objects()

            if self._add_bg:
                self.add_bg()
            else:
                self._world.scene.add_default_ground_plane()

            if self._add_rep:
                self.add_bg_repliactor()
                self.add_manipulator_replicator()
            if self._collect_robo_data:
                self.setup_dataset()

        if self._add_rep:
            self.setup_replicator()
        return

    def create_materials(self):
        return

    def add_bg_property(self):
        prim_path = f"/World/Env_{self._index}/background_{self._index}"
        
        from omni.isaac.etri_bridge import add_physical_properties
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=False,
            collision_approximation="meshSimplification"
        )
        return

    def add_objects_physical_properties(self):
        return

    def create_dataset(self):
        return

    def setup_post_load(self):
        self.create_materials()

        for i in range(len(self._env_offsets)):
            self._index = i
            if self._add_bg:
                self.add_bg_property()
            
            self.create_dataset()
            self.add_objects_physical_properties()
        return

    def forward(self, save_count):
        return