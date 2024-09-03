# Pick and Place Franka Version

# from omni.isaac.nucleus import get_assets_root_path, get_url_root
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.prims.geometry_prim import GeometryPrim
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.sensor import Camera

import omni.replicator.core as rep
from os.path import expanduser
import omni.graph.core as og
import numpy as np
import omni.usd
import omni

from omni.isaac.custom_controller import CustomPickPlaceController
from omni.isaac.robo_data_collector import RoboDataCollector
from omni.isaac.franka import Franka

from ..multi_env_utils import create_aluminum_polished_material, create_iron_material
from ..multi_env_utils import create_copper_material, create_glass_material
from ..multi_env_utils import add_physical_properties, add_material_to_prim
from ..multi_env_utils import add_object, add_semantic_data
from ..multi_env import MultiEnv


class MultiRoboticRack(MultiEnv):
    def __init__(
            self,
            world=None,
            env_offsets: list = [0.0, 0.0, 0.0],
            add_background: bool = False,
            bg_path: str = None,
            add_replicator: bool = False,
            verbose: bool = False,
            robot_type: str = "2f85", # 2f85, 3f
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

        self.Tanir_Makina_Factory_Building_PATH = self._server_root + "/Projects/ETRI/Env/RobotRack/Tanır_Makina_Factory_Building.usdz"
        self.Wooden_Industrial_Shelf_PATH = self._server_root + "/Projects/ETRI/Env/RobotRack/Wooden_Industrial_Shelf.usdz"
        self.Worn_Warehouse_Shelf_PATH = self._server_root + "/Projects/ETRI/Env/RobotRack/Worn_Warehouse_Shelf.usdz"
        self.Industrial_Locker_PATH = self._server_root + "/Projects/ETRI/Env/RobotRack/Industrial_-_Locker.usdz"
        
        self.RampLong_A1_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Railing/RampLong_A1.usd"
        self.MetalFencing_A2_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Railing/MetalFencing_A2.usd"
        self.MetalFencing_A3_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Railing/MetalFencing_A3.usd"
        self.RackLong_A1_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Racks/RackLong_A1.usd"
        self.RackLongEmpty_A2_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Shelves/RackLongEmpty_A2.usd"
        self.WarehousePile_A3_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Piles/WarehousePile_A3.usd"
        self.Cardbox_A2_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A2.usd"
        self.CornerRail_A1_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Railing/CornerRail_A1.usd"

        self.Industrial_Table_PATH = self._server_root + "/Projects/ETRI/Env/RobotRack/Industrial_Table.usdz"
        self.Industrial_steel_table_PATH = self._server_root + "/Projects/ETRI/Env/RobotRack/Industrial_steel_table.usdz"
        self.Cardbox_C1_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_C1.usd"
        self.Cardbox_A3_URL = self._isaac_assets_path + "/NVIDIA/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A3.usd"
        self.Wine_Glass_PATH = self._server_root + "/Projects/ETRI/Glass/Wine_Glass.usdz"
        
        self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/Franka/franka_realsense.usd"
        
        self._bg_scale = (1, 1, 1)
        self._bg_translation = (0, 0, 0)
        self._bg_orientation = (1, 0, 0, 0)

        self._robot_position = np.array([0.4, 0.5, 0.66])
        self._robot_orientation = np.array([0.7071068, 0, 0, -0.7071068])
        return

############################

    def add_wooden_industrial_shelf(self):
        #[Wooden Industrial Shelf]
        prim_path = f"/World/Env_{self._index}/Wooden_Industrial_Shelf_{self._index}"
        add_object(
            usd_path=self.Wooden_Industrial_Shelf_PATH,
            prim_path=prim_path,
            translate=(7.5, 8.1, 1.0),
            orient=(0.7071068, 0.7071068, 0, 0),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "wooden_industrial_shelf")
        
    def add_worn_warehouse_shelf(self):
        #[Worn Warehouse Shelf]
        prim_path = f"/World/Env_{self._index}/Worn_Warehouse_Shelf_{self._index}"
        add_object(
            usd_path=self.Worn_Warehouse_Shelf_PATH,
            prim_path=prim_path,
            translate=(7.0, -6.6, 0.0),
            orient=(0, 0, 0.7071068, 0.7071068),
            scale=(0.008, 0.008, 0.008),
        )
        add_semantic_data(prim_path, "worn_warehouse_shelf")

    def add_industrial_locker(self):
        #[Industrial Locker]
        prim_path = f"/World/Env_{self._index}/Industrial_Locker_{self._index}"
        add_object(
            usd_path=self.Industrial_Locker_PATH,
            prim_path=prim_path,
            translate=(-4.5, 8.0, 1.55),
            orient=(0.7071068, 0.7071068, 0, 0),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "industrial_locker")

    def add_ramplong_a1(self):
        #[RampLong_A1]
        prim_path = f"/World/Env_{self._index}/RampLong_A1_{self._index}"
        add_object(
            usd_path=self.RampLong_A1_URL,
            prim_path=prim_path,
            translate=(0.0, -2.0, 0.0),
            orient=(1, 0, 0, 0),
            scale=(0.03, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "ramp_long")


    def add_metal_fencing_a2(self):
        #[MetalFencing_A2]
        RampShort_A2_list = ["RampShort_A2_01", "RampShort_A2_02", "RampShort_A2_03", "RampShort_A2_04", "RampShort_A2_05", "RampShort_A2_06"]
        for i in range(0,3):
            prim_path = f"/World/Env_{self._index}/{RampShort_A2_list[i]}_{self._index}"
            add_object(
                usd_path=self.MetalFencing_A2_URL,
                prim_path=prim_path,
                translate=(-3.3, -1.5+(i*1.5), 0.0),
                orient=(0.7071068, 0, 0, 0.7071068),
                scale=(0.01, 0.01, 0.01),
            )
            add_semantic_data(prim_path, f"rampshort_{i}")

        for i in range(3,6):
            prim_path = f"/World/Env_{self._index}/{RampShort_A2_list[i]}_{self._index}"
            add_object(
                usd_path=self.MetalFencing_A2_URL,
                prim_path=prim_path,
                translate=(-2.2 + (i-3)*1.6, 3.5, 0),
                orient=(1, 0, 0, 0),
                scale=(0.01, 0.01, 0.01),
            )
            add_semantic_data(prim_path, f"rampshort_{i}")

    def add_metal_fencing_a3(self):
        #[MetalFencing_A3]
        prim_path = f"/World/Env_{self._index}/MetalFencing_A3_01_{self._index}"
        add_object(
            usd_path=self.MetalFencing_A3_URL,
            prim_path=prim_path,
            translate=(-3.3, 3.0, 0.0),
            orient=(1, 0, 0, 0),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "metal_fencing_a3")

        prim_path = f"/World/Env_{self._index}/MetalFencing_A3_02_{self._index}"
        add_object(
            usd_path=self.MetalFencing_A3_URL,
            prim_path=prim_path,
            translate=(2.6, 3.5, 0),
            orient=(0.7071068, 0, 0, -0.7071068),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "metal_fencing_a3")


    def add_racklong_a1(self):
        #[RackLong_A1]
        prim_path = f"/World/Env_{self._index}/RackLong_A1_{self._index}"
        add_object(
            usd_path=self.RackLong_A1_URL,
            prim_path=prim_path,
            translate=(2.0, 0.0, 0.0),
            orient=(0, 0, 0, 1),
            scale=(0.005, 0.005, 0.005),
        )
        add_semantic_data(prim_path, "racklong")

    def add_racklong_empty_a2(self):
        #[RackLongEmpty_A2]
        prim_path = f"/World/Env_{self._index}/RackLongEmpty_A2_{self._index}"
        add_object(
            usd_path=self.RackLongEmpty_A2_URL,
            prim_path=prim_path,
            translate=(-1, 2, 0),
            orient=(0.7071068, 0, 0, 0.7071068),
            scale=(0.005, 0.005, 0.005),
        )
        add_semantic_data(prim_path, "racklong_empty")

    def add_warehouse_pile_a3(self):
        #[WarehousePile_A3]
        prim_path = f"/World/Env_{self._index}/WarehousePile_A3_{self._index}"
        add_object(
            usd_path=self.WarehousePile_A3_URL,
            prim_path=prim_path,
            translate=(-1.8, 0, 0),
            orient=(0.258819, 0, 0, -0.9659258),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "warehouse_pile")

    def add_corner_rail_a1(self):
        #[CornerRail_A1]
        prim_path = f"/World/Env_{self._index}/CornerRail_A1_{self._index}"
        add_object(
            usd_path=self.CornerRail_A1_URL,
            prim_path=prim_path,
            translate=(2.4, 1.0, 0),
            orient=(0.7071068, 0, 0, 0.7071068),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "corner_rail_a1")

############################

    def add_bg(self):
        prim_path = f"/World/Env_{self._index}/background_{self._index}"

        add_object(
            usd_path=self.BG_PATH,
            prim_path=prim_path,
            scale=self._bg_scale,
            translate=self._bg_translation,
            orient=self._bg_orientation,
        )
        add_semantic_data(prim_path, "background")
        return

    def add_camera(self):
        prim_path = f"/World/Env_{self._index}/front_camera"
        self._front_camera = Camera(
            prim_path=prim_path,
            position=np.array([0.4, -1.5, 1.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=np.array([
                0.7071068, 0, 0, 0.7071068, 
            ]), 
        )
        self._front_camera.set_focal_length(1.5)
        self._front_camera.initialize()
        self._front_camera.add_motion_vectors_to_frame()

        prim_path = f"/World/Env_{self._index}/top_camera"
        self._top_camera = Camera(
            prim_path=prim_path,
            position=np.array([0.4, 0.3, 3.0]),
            frequency=30,
            resolution=(640, 480),
            # TODO : omni.isaac.core.utils.rotations
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0, 90, 90
                ]), degrees=True),
        )
        self._top_camera.set_focal_length(1.5)
        self._top_camera.initialize()
        self._top_camera.add_motion_vectors_to_frame()

        prim_path = f"/World/Env_{self._index}/side_camera"
        self._side_camera = Camera(
            prim_path=prim_path,
            position=np.array([1.8, 0.0, 1.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0, 0, 180
                ]), degrees=True),
        )
        self._side_camera.set_focal_length(1.5)
        self._side_camera.initialize()
        self._side_camera.add_motion_vectors_to_frame()

    def add_bg_repliactor(self):
        prim_path = f"/World/Env_{self._index}/front_camera"
        self._rp_front = rep.create.render_product(
            prim_path,
            (640, 480),
            name="front_camera",
            force_new=False ,
        )

        prim_path = f"/World/Env_{self._index}/top_camera"
        self._rp_top = rep.create.render_product(
            prim_path,
            (640, 480),
            name="top_camera",
            force_new=False ,
        )
        
        prim_path = f"/World/Env_{self._index}/side_camera"
        self._rp_side = rep.create.render_product(
            prim_path,
            (640, 480),
            name="side_camera",
            force_new=False ,
        )

    def add_manipulator_replicator(self):
        prim_path = f"/World/Env_{self._index}/Franka_{self._index}/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
        self._rp_ee = rep.create.render_product(
            prim_path,
            (640, 480),
            name="ee_camera",
            force_new=False ,
        )

    def add_og(self):

        sub_topic_name = f"/issac/joint_states"
        pub_topic_name = f"/issac/joint_command"
        prim_path = f"/World/Env_0/robot_0"

        try:
            # Clock OG
            og.Controller.edit(
                {"graph_path": "/ROS2Clock", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("context", "omni.isaac.ros2_bridge.ROS2Context"),
                        ("readSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("publishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "publishClock.inputs:execIn"),
                        ("readSimTime.outputs:simulationTime", "publishClock.inputs:timeStamp"),
                        ("context.outputs:context", "publishClock.inputs:context"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        # ("context.inputs:domain_id", domain_id),
                        ("readSimTime.inputs:resetOnStop", True),
                    ],
                },
            )

            # Joint Pub Sub
            og.Controller.edit(
                {"graph_path": "/ROS2JointPubSub", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("onPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                        ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                        ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                        ("readSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("onPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                        ("onPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                        ("onPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                        ("readSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                        ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                        ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                        ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                        ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("ArticulationController.inputs:usePath", False),
                        ("ArticulationController.inputs:targetPrim", prim_path),
                        ("PublishJointState.inputs:targetPrim", prim_path),
                        ("SubscribeJointState.inputs:topicName", pub_topic_name),
                        ("PublishJointState.inputs:topicName", sub_topic_name),
                        ("readSimTime.inputs:resetOnStop", True),
                    ],
                },
            )

        except Exception as e:
            print(e)

    def add_light(self):
        # Create a sphere light in Industrial unit
        sphereLight = UsdLux.SphereLight.Define(self._stage, Sdf.Path(f"/World/Env_{self._index}/sphereLight_{self._index}"))
        sphereLight.CreateIntensityAttr(300000)
        translate = (0.0, 0.0, 7.0)
        tx, ty, tz = tuple(translate)
        sphereLight.AddRotateXYZOp().Set((0, 0, 0))
        sphereLight.AddTranslateOp().Set(Gf.Vec3f(tx, ty, tz))

    def add_robot(self):
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
            end_effector_initial_height=1.3,
            robot_type="franka",
        )
        
        self._controllers.append(controller)
        return

    def add_industrial_table_1(self):
        # Industrial_Table_1
        prim_path = f"/World/Env_{self._index}/Industrial_Table_{self._index}"
        add_object(
            usd_path=self.Industrial_Table_PATH,
            prim_path=prim_path,
            translate=(0.4, 0.6, 0.0),
            orient=(0.7071068, 0.7071068, 0, 0),
            scale=(0.008, 0.008, 0.008),
        )
        add_semantic_data(prim_path, "industrial_table")

    def add_industrial_steel_table_1(self):
        # Industrial_steel_table_1
        prim_path = f"/World/Env_{self._index}/Industrial_steel_table_{self._index}"
        add_object(
            usd_path=self.Industrial_steel_table_PATH,
            prim_path=prim_path,
            translate=(0.4, -0.1, 0.0),
            orient=(0.5, 0.5, 0.5, 0.5),
            scale=(0.002, 0.002, 0.002),
        )
        add_semantic_data(prim_path, "industrial_steel_table")

    def add_cardbox(self):
        #[Cardbox_C1]
        prim_path = f"/World/Env_{self._index}/Cardbox_C1_{self._index}"
        add_object(
            usd_path=self.Cardbox_C1_URL,
            prim_path=prim_path,
            translate=(-0.35, 0.6, 0.5),
            orient=(1, 0, 0, 0),
            scale=(0.01, 0.01, 0.02),
        )
        add_semantic_data(prim_path, "cardbox_c1")

        #[Cardbox_A1]
        prim_path = f"/World/Env_{self._index}/Cardbox_A1_{self._index}"
        add_object(
            usd_path=self.Cardbox_A3_URL,
            prim_path=prim_path,
            translate=(-0.35, 0.6, -0.01365),
            orient=(1, 0, 0, 0),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "cardbox_a1")

    def add_cardbox2(self):
        #[Cardbox_C2]
        prim_path = f"/World/Env_{self._index}/Cardbox_C2_{self._index}"
        add_object(
            usd_path=self.Cardbox_C1_URL,
            prim_path=prim_path,
            translate=(-0.56, 2.0, 0.26),
            orient=(1, 0, 0, 0),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "cardbox_c2")

        #[Cardbox_A2]
        prim_path = f"/World/Env_{self._index}/Cardbox_A2_{self._index}"
        add_object(
            usd_path=self.Cardbox_A3_URL,
            prim_path=prim_path,
            translate=(-1.4, 2.0, 1.15),
            orient=(1, 0, 0, 0),
            scale=(0.01, 0.01, 0.01),
        ) 
        add_semantic_data(prim_path, "cardbox_a2")

        #[Cardbox_A3]
        prim_path = f"/World/Env_{self._index}/Cardbox_A3_{self._index}"
        add_object(
            usd_path=self.Cardbox_A2_URL,
            prim_path=prim_path,
            translate=(-0.5, 2.0, 1.15),
            orient=(1, 0, 0, 0),
            scale=(0.01, 0.01, 0.01),
        )
        add_semantic_data(prim_path, "cardbox_a3")


    def add_wine_glass(self):
        # wine glass
        # x, y, z = np.array([-0.2, 0.6, 1.07])
        # prim_path = f"/World/Env_{self._index}/Wine_Glass_1_{self._index}"
        # add_object(
        #     usd_path=self.Wine_Glass_PATH,
        #     prim_path=prim_path,
        #     translate=(x, y, z),
        #     orient=(0.5, 0.5, 0.5, 0.5),
        #     scale=(0.001, 0.001, 0.001),
        # )
        # add_semantic_data(prim_path, "wine_glass")

        x, y, z = np.array([-0.2, 0.75, 1.07])
        prim_path = f"/World/Env_{self._index}/Wine_Glass_1_{self._index}"
        add_object(
            usd_path=self.Wine_Glass_PATH,
            prim_path=prim_path,
            translate=(x, y, z),
            orient=(0.5, 0.5, 0.5, 0.5),
            scale=(0.001, 0.001, 0.001),
        )
        add_semantic_data(prim_path, "wine_glass")

    def add_objects(self):
        self.add_industrial_table_1()
        self.add_industrial_steel_table_1()
        self.add_cardbox()
        self.add_wine_glass()

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

    def setup_replicator(self):
        from os.path import expanduser
        self._out_dir = str(expanduser("~") + "/Documents/ETRI_DATA/robot_pp_" + self._now_str)

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

    def setup_dataset(self):
        self._robo_data_collector = RoboDataCollector(file_name=f"robot{self._index}_pp_data_" + self._now_str)
        self._save_count = 0

        self._robo_data_collectors.append(self._robo_data_collector)
        return

    def setup_scene(self):
        self._stage = omni.usd.get_context().get_stage()

        if self._add_bg is False:
            # add defalut grond plane은 한번만 하면 됨
            self._world.scene.add_default_ground_plane()
            # Turn off the default light
            self.turn_off_dg_light()

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
                # 추신 - 카메라 추가하면 급격히 느려짐
                # self.add_camera()
                self.add_bg_repliactor()
                self.add_manipulator_replicator()
                self.setup_replicator()
            if self._collect_robo_data:
                self.setup_dataset()

            if self._verbose:
                print(f"[MultiRoboticRack] Env_{self._index} Loaded")
        return

    def create_materials(self):
        # Aluminum_Polished
        self._aluminum_polished_material = create_aluminum_polished_material(self._stage)

        # Copper
        self._copper_material = create_copper_material(self._stage)

        # Iron
        self._iron_material = create_iron_material(self._stage)
        
        # Glass
        self._glass_material = create_glass_material(self._stage)

    def add_bg_property(self):
        prim_path = f"/World/Env_{self._index}/background_{self._index}"
        
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=False,
            collision_approximation="meshSimplification"
        )

    def add_industrial_table_1_property(self):
        prim_path = f"/World/Env_{self._index}/Industrial_Table_{self._index}"
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=True,
            collision_approximation="meshSimplification"
        )

    def add_industrial_steel_table_1_property(self):
        prim_path = f"/World/Env_{self._index}/Industrial_steel_table_{self._index}"
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=True,
            collision_approximation="convexDecomposition"
        )
        add_material_to_prim(prim_path, self._aluminum_polished_material)

    def add_cardbox_property(self):
        prim_path = f"/World/Env_{self._index}/Cardbox_C1_{self._index}"
        Cardbox_C1_geom, Cardbox_C1_rigid = add_physical_properties(
            prim_path=prim_path,
            rigid_body=True,
            collision_approximation="convexDecomposition"
        )
        Cardbox_C1_rigid.CreateKinematicEnabledAttr(True)

        prim_path = f"/World/Env_{self._index}/Cardbox_A1_{self._index}"
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=True,
            collision_approximation="meshSimplification"
        )

    def add_wine_glass_property(self):
        prim_path = f"/World/Env_{self._index}/Wine_Glass_1_{self._index}"
        Wine_Glass_1_geom, _ = add_physical_properties(
            prim_path=prim_path,
            rigid_body=True,
            collision_approximation="convexDecomposition"
        )
        self._world.scene.add(Wine_Glass_1_geom)
        add_material_to_prim(prim_path, self._glass_material)

        # prim_path = f"/World/Env_{self._index}/Wine_Glass_2_{self._index}"
        # Wine_Glass_2_geom, _ = add_physical_properties(
        #     prim_path=prim_path,
        #     rigid_body=True,
        #     collision_approximation="convexDecomposition"
        # )
        # self._world.scene.add(Wine_Glass_2_geom)
        # add_material_to_prim(prim_path, self._glass_material)

    def add_objects_physical_properties(self):
        self.add_industrial_table_1_property()
        self.add_industrial_steel_table_1_property()
        self.add_cardbox_property()
        self.add_wine_glass_property()

        if self._add_bg is True:
            self.add_wooden_industrial_shelf_property()
            self.add_worn_warehouse_shelf_property()
            self.add_industrial_locker_property()
            self.add_ramplong_a1_property()
            self.add_metal_fencing_property()
            self.add_racklong_property()
            self.add_warehouse_pile_a3_property()
            self.add_corner_rail_a1_property()

    def setup_robots(self):
        robot = self._world.scene.get_object(f"robot_{self._index}")
        robot.set_joints_default_state(positions=np.zeros(9))
        robot.gripper.set_joint_positions(robot.gripper.joint_opened_positions)
        self._robots.append(robot)

    def create_dataset(self):
        robot = self._world.scene.get_object(f"robot_{self._index}")
        print(f"self._index : {self._index}")
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
    
    def set_motion_dict(self, motion_dict):
        self._motion_dict = motion_dict
        self._end_effector_offset = self._motion_dict["end_effector_offset"]
        self._object_name = self._motion_dict["object_name"]

        self._picking_offset = self._motion_dict["picking_offset"]
        self._picking_object_offset = self._motion_dict["picking_object_offset"]
        self._picking_orientation_local = self._motion_dict["picking_orientation"]
        self._picking_orientation_world = rot_matrix_to_quat(
            quat_to_rot_matrix(self._robot_orientation) @ quat_to_rot_matrix(self._picking_orientation_local)
        )

        self._placing_position = self._motion_dict["placing_position"]
        self._placing_offset = self._motion_dict["placing_offset"]
        self._placing_object_offset = self._motion_dict["placing_object_offset"]
        self._placing_orientation_local = self._motion_dict["placing_orientation"]
        self._placing_orientation_world = rot_matrix_to_quat(
            quat_to_rot_matrix(self._robot_orientation) @ quat_to_rot_matrix(self._placing_orientation_local)
        )

    def forward(self):

        for i, robot in enumerate(self._robots):

            object_geom = self._world.scene.get_object(f"{self._object_name}_{i}")
            object_position, _ = object_geom.get_world_pose()
            print(f"Object Position: {object_position}")

            env_offset = self._env_offsets[i]
            picking_position = object_position + self._picking_object_offset
            placing_position = self._placing_position + self._placing_object_offset + env_offset

            actions = self._controllers[i].forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=robot.get_joint_positions(),
                picking_offset=self._picking_offset,
                placing_offset=self._placing_offset,
                end_effector_offset=self._end_effector_offset,
                picking_end_effector_orientation=self._picking_orientation_world,
                placing_end_effector_orientation=self._placing_orientation_world,
            )
            if self._controllers[i].is_done():
                print(f"[Env {i}] done picking and placing")
                # self._controllers[i].reset()
            else:
                robot.apply_action(actions)

            if self._collect_robo_data:
                self._robo_data_collectors[i].create_single_pp_data(
                    name=f"step_{self._save_count}",
                    robot=robot,
                    controller=self._controllers[i],
                )

        self._save_count += 1

##########################################

    def add_wooden_industrial_shelf_property(self):
        Wooden_Industrial_Shelf_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/Wooden_Industrial_Shelf_{self._index}"))
        Wooden_Industrial_Shelf_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/Wooden_Industrial_Shelf_{self._index}", name=f"Wooden_Industrial_Shelf_{self._index}", collision=True)
        Wooden_Industrial_Shelf_geom.set_collision_approximation("convexDecomposition")

    def add_worn_warehouse_shelf_property(self):
        Worn_Warehouse_Shelf_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/Worn_Warehouse_Shelf_{self._index}"))
        Worn_Warehouse_Shelf_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/Worn_Warehouse_Shelf_{self._index}", name=f"Worn_Warehouse_Shelf_{self._index}", collision=True)
        Worn_Warehouse_Shelf_geom.set_collision_approximation("convexDecomposition")

    def add_industrial_locker_property(self):
        Industrial_Locker_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/Industrial_Locker_{self._index}"))
        Industrial_Locker_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/Industrial_Locker_{self._index}", name=f"Industrial_Locker_{self._index}", collision=True)
        Industrial_Locker_geom.set_collision_approximation("convexDecomposition")

    def add_ramplong_a1_property(self):
        RampLong_A1_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/RampLong_A1_{self._index}"))
        RampLong_A1_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/RampLong_A1_{self._index}", name=f"RampLong_A1_{self._index}", collision=True)
        RampLong_A1_geom.set_collision_approximation("convexDecomposition")

    def add_metal_fencing_property(self):
        MetalFencing_A2_list = ["RampShort_A2_01", "RampShort_A2_02", "RampShort_A2_03", "RampShort_A2_04", "RampShort_A2_05", "RampShort_A2_06"]
        for i in range(0,3):
            MetalFencing_A2_list_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/{MetalFencing_A2_list[i]}_{self._index}"))
            MetalFencing_A2_list_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/{MetalFencing_A2_list[i]}_{self._index}", name=f"{MetalFencing_A2_list[i]}_{self._index}", collision=True)
            MetalFencing_A2_list_geom.set_collision_approximation("convexDecomposition")

        for i in range(3,6):
            MetalFencing_A2_list_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/{MetalFencing_A2_list[i]}_{self._index}"))
            MetalFencing_A2_list_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/{MetalFencing_A2_list[i]}_{self._index}", name=f"{MetalFencing_A2_list[i]}_{self._index}", collision=True)
            MetalFencing_A2_list_geom.set_collision_approximation("convexDecomposition")

        MetalFencing_A3_01_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/MetalFencing_A3_01_{self._index}"))
        MetalFencing_A3_01_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/MetalFencing_A3_01_{self._index}", name=f"MetalFencing_A3_01_{self._index}", collision=True)
        MetalFencing_A3_01_geom.set_collision_approximation("convexDecomposition")

        MetalFencing_A3_02_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/MetalFencing_A3_02_{self._index}"))
        MetalFencing_A3_02_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/MetalFencing_A3_02_{self._index}", name=f"MetalFencing_A3_02_{self._index}", collision=True)
        MetalFencing_A3_02_geom.set_collision_approximation("convexDecomposition")

    def add_racklong_property(self):
        RackLong_A1_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/RackLong_A1_{self._index}"))
        RackLong_A1_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/RackLong_A1_{self._index}", name=f"RackLong_A1_{self._index}", collision=True)
        RackLong_A1_geom.set_collision_approximation("convexDecomposition")

        RackLongEmpty_A2_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/RackLongEmpty_A2_{self._index}"))
        RackLongEmpty_A2_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/RackLongEmpty_A2_{self._index}", name=f"RackLongEmpty_A2_{self._index}", collision=True)
        RackLongEmpty_A2_geom.set_collision_approximation("convexDecomposition")

    def add_warehouse_pile_a3_property(self):
        WarehousePile_A3_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/WarehousePile_A3_{self._index}"))
        WarehousePile_A3_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/WarehousePile_A3_{self._index}", name=f"WarehousePile_A3_{self._index}", collision=True)
        WarehousePile_A3_geom.set_collision_approximation("convexDecomposition")

    def add_corner_rail_a1_property(self):
        CornerRail_A1_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/CornerRail_A1_{self._index}"))
        CornerRail_A1_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/CornerRail_A1_{self._index}", name=f"CornerRail_A1_{self._index}", collision=True)
        CornerRail_A1_geom.set_collision_approximation("convexDecomposition")

    def add_metal_bucket_property(self):
        Metal_Bucket_rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(f"/World/Env_{self._index}/Metal_Bucket_{self._index}"))
        Metal_Bucket_geom = GeometryPrim(prim_path=f"/World/Env_{self._index}/Metal_Bucket_{self._index}", name=f"Metal_Bucket_{self._index}", collision=True)
        Metal_Bucket_geom.set_collision_approximation("convexDecomposition")
