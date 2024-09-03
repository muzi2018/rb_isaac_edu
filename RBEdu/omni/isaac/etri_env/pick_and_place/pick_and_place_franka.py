# Pick and Place Franka Version
import omni.replicator.core as rep
import omni.graph.core as og
import numpy as np
import omni.usd
import omni

from omni.isaac.custom_franka import Franka2F, Franka3F

from omni.isaac.custom_controller import CustomPickPlaceController
from omni.isaac.robo_data_collector import RoboDataCollector
from omni.isaac.franka import Franka

from ..multi_env_utils import add_physical_properties, add_material_to_prim
from ..multi_env_utils import add_object, add_semantic_data
from .pick_and_place_env import MultiRoboticRack


class MultiRoboticRackFranka(MultiRoboticRack):
    def __init__(
            self,
            world=None,
            env_offsets: list = [0.0, 0.0, 0.0],
            add_background: bool = False,
            bg_path: str = None,
            add_replicator: bool = False,
            verbose: bool = False,
            robot_type: str = "basic", # 2f85, 3f
            add_ros2: bool = False,
            collect_robo_data: bool = False,
        ) -> None:

        super().__init__(
            world, 
            env_offsets, 
            add_background, 
            bg_path, 
            add_replicator, 
            verbose, 
            robot_type, 
            add_ros2, 
            collect_robo_data
        )

        if self._robot_type == "basic":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/Franka/franka_realsense.usd"
        elif self._robot_type == "2f85":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/Franka/franka_2f85.usd"
        elif self._robot_type == "3f":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/Franka/franka_3f.usd"

        self.GLASS_CUP_PATH = self._server_root + "/Projects/ETRI/Glass/Glass_cup.usdz"

        self._bg_scale = (0.03, 0.03, 0.03)
        self._bg_translation = (9.5, -7.5, -0.22)
        self._bg_orientation = (0.5, 0.5, 0.5, 0.5)
        
        self._robot_position = np.array([0.4, 0.5, 0.66])
        self._robot_orientation = np.array([0.7071068, 0, 0, -0.7071068])
        return

    def add_robot(self):

        robot = None

        if self._robot_type == "2f85":
            robot = self._world.scene.add(
                Franka2F(
                    prim_path=f"/World/Env_{self._index}/robot_{self._index}",
                    usd_path=self.ROBOT_PATH,
                    name=f"robot_{self._index}",
                    position=self._robot_position + self._xyz_offset,
                    orientation=self._robot_orientation,
                    attach_gripper=True,
                )
            )
        elif self._robot_type == "3f":
            robot = self._world.scene.add(
                Franka3F(
                    prim_path=f"/World/Env_{self._index}/robot_{self._index}",
                    usd_path=self.ROBOT_PATH,
                    name=f"robot_{self._index}",
                    position=self._robot_position + self._xyz_offset,
                    orientation=self._robot_orientation,
                    attach_gripper=True,
                )
            )
        elif self._robot_type == "basic":
            robot = self._world.scene.add(
                Franka(
                    usd_path=self.ROBOT_PATH,
                    prim_path=f"/World/Env_{self._index}/robot_{self._index}",
                    name=f"robot_{self._index}",
                    position=self._robot_position + self._xyz_offset,
                    orientation=self._robot_orientation,
                )
            )

        controller = CustomPickPlaceController(
            name="pick_place_controller",
            gripper=robot.gripper,
            robot_articulation=robot,
            end_effector_initial_height=1.2,
            robot_type="franka",
        )
        
        self._controllers.append(controller)

    def add_glass_cup(self):
        # Glass Cup
        prim_path = f"/World/Env_{self._index}/Glass_Cup_1_{self._index}"
        add_object(
            usd_path=self.GLASS_CUP_PATH,
            prim_path=prim_path,
            translate=(-0.17, 0.75, 1.1),
            orient=(0.7071068, 0.7071068, 0, 0),
            scale=(0.002, 0.002, 0.002),
        )
        add_semantic_data(prim_path, "glass_cup")
        return

    # Custom Parts
    def add_objects(self):
        self.add_industrial_table_1()
        self.add_industrial_steel_table_1()
        self.add_cardbox()
        # self.add_wine_glass()
        self.add_glass_cup()

        if self._add_bg is True:
            self.add_wooden_industrial_shelf()
            self.add_worn_warehouse_shelf()
            self.add_industrial_locker()
            self.add_ramplong_a1()
            self.add_metal_fencing_a2()
            self.add_metal_fencing_a3()
            self.add_racklong_a1()
            self.add_racklong_empty_a2()
            self.add_warehouse_pile_a3()
            self.add_corner_rail_a1()
            self.add_cardbox2()

    def add_manipulator_replicator(self):
        if self._robot_type == "basic":
            prim_path = f"/World/Env_{self._index}/robot_{self._index}/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
        elif self._robot_type == "2f85":
            prim_path = f"/World/Env_{self._index}/robot_{self._index}/_f85_basic/robotiq_arg2f_base_link/Realsense/Realsense/RSD455/Camera_OmniVision_OV9782_Color"

        self._rp_ee = rep.create.render_product(
            prim_path,
            (640, 480),
            name="ee_camera",
            force_new=False ,
        )

    def setup_dataset(self):
        self._robo_data_collector = RoboDataCollector(file_name=f"franka_2f_pp_env{self._index}_pp_data_" + self._now_str)
        self._save_count = 0
        self._robo_data_collectors.append(self._robo_data_collector)
        return

    def setup_replicator(self):
        from os.path import expanduser
        self._out_dir = str(expanduser("~") + f"/Documents/ETRI_DATA/franka_2f_pp_env{self._index}_" + self._now_str)

        self._writer = rep.WriterRegistry.get("CustomBasicWriter")
        self._writer.initialize(
            output_dir=self._out_dir, 
            rgb=True,
            # bounding_box_2d_tight=True,
            # bounding_box_2d_loose=True,
            semantic_segmentation=True,
            # instance_id_segmentation=True,
            # instance_segmentation=True,
            bounding_box_3d=True,
            distance_to_camera=True,
            # pointcloud=True,
            # colorize_semantic_segmentation=True,
            # colorize_instance_id_segmentation=True,
            # colorize_instance_segmentation=True,
        )

        self._render_products = [
            self._rp_ee,
            self._rp_top,
            self._rp_front,
            self._rp_side
        ]
        self._writer.attach(self._render_products)

    def setup_scene(self):
        self._stage = omni.usd.get_context().get_stage()

        if self._verbose:
            print(f"self._add_bg: {self._add_bg}")

        if self._add_bg is False:
            # add defalut grond plane은 한번만 하면 됨
            self._world.scene.add_default_ground_plane()
            # Turn off the default light
            self.turn_off_dg_light()

        # ros2 og는 단 하나의 로봇에만 해당 
        if self._add_ros2:
            self.add_og()

        for i in range(self._num_lens):
            self._index = i
            self._xyz_offset = np.array(self._env_offsets[self._index])

            if self._verbose:
                print(f"[MultiRoboticRack] Loading Env_{self._index} / xyz_offset: {self._xyz_offset}")
            
            self.add_xform()
            self.add_light()
            self.add_robot()
            self.add_objects()

            # TODO: 배경도 추가해보자. 아직 구현 안해두었음
            # 배경을 하나의 usd로 만들어서 추가하면 편할 것 같음
            if self._add_bg:
                self.add_bg()
            if self._add_rep:
                self.add_camera()
                self.add_bg_repliactor()
                self.add_manipulator_replicator()
                self.setup_replicator()
            if self._collect_robo_data:
                self.setup_dataset()

            if self._verbose:
                print(f"[MultiRoboticRack] Env_{self._index} Loaded")

        return

    def setup_robots(self):
        robot = self._world.scene.get_object(f"robot_{self._index}")
        if self._robot_type == "2f85" or self._robot_type == "basic":
            robot.set_joints_default_state(positions=np.zeros(9))
            robot.gripper.set_joint_positions(robot.gripper.joint_opened_positions)
        elif self._robot_type == "3f":
            robot.set_joints_default_state(positions=np.zeros(11))
        self._robots.append(robot)

    def add_glass_cup_property(self):
        prim_path = f"/World/Env_{self._index}/Glass_Cup_1_{self._index}"
        Glass_Cup_1_geom, _ = add_physical_properties(
            prim_path=prim_path,
            rigid_body=True,
            collision_approximation="convexDecomposition"
        )
        self._world.scene.add(Glass_Cup_1_geom)
        add_material_to_prim(prim_path, self._glass_material)

    def add_objects_physical_properties(self):
        self.add_industrial_table_1_property()
        self.add_industrial_steel_table_1_property()
        self.add_cardbox_property()
        # self.add_wine_glass_property()
        self.add_glass_cup_property()

    def create_dataset(self):
        robot = self._world.scene.get_object(f"robot_{self._index}")
        robo_data_collector = self._robo_data_collectors[self._index]

        joint_names = robot._articulation_view.joint_names
        robo_data_collector.create_dataset(
            "joint_names",
            {
                "franka_emika": joint_names,
            }
        )

    def setup_post_load(self):
        self._world.scene.enable_bounding_boxes_computations()
        self.create_materials()

        for i in range(self._num_lens):
            self._index = i
            if self._add_bg:
                self.add_bg_property()
            if self._collect_robo_data:
                self.create_dataset()
            self.setup_robots()
            self.add_objects_physical_properties()
        return
    
    def setup_pre_reset(self):
        if self._collect_robo_data:
            for robo_data_collector in self._robo_data_collectors:
                robo_data_collector.setup_pre_reset()
        self._save_count = 0
        return
    
    def world_cleanup(self):
        if self._collect_robo_data:
            for robo_data_collector in self._robo_data_collectors:
                robo_data_collector.save_data()
        self._save_count = 0
        return