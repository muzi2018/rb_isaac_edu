from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema
# from omni.isaac.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.prims.xform_prim import XFormPrim

import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from os.path import expanduser
import numpy as np
import omni.usd
import omni

from omni.isaac.ur5 import UR52F, UR53F
from omni.isaac.custom_controller import NutBoltController 
from omni.isaac.robo_data_collector import RoboDataCollector

from .screw_env import MultiScrewEnv
from ..multi_env_utils import add_object, add_semantic_data, add_physical_properties


class MultiScrewEnvUR5(MultiScrewEnv):
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

        # pipe and bolt parameters
        self._num_nuts = 2
        self._nut_height = 0.016
        self._num_bolts = 6
        self._bolt_length = 0.1
        self._gripper_to_nut_offset = np.array([0.0, 0.0, 0.16])
        self._gripper_to_bolt_offset = np.array([0.0, 0.0, 0.15])
        self._top_of_bolt = (
            np.array([0.001, 0.0, self._bolt_length + (self._nut_height / 2)]) + self._gripper_to_bolt_offset
        )

        if self._robot_type == "2f85":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR5/ur5_2f85.usd"
        elif self._robot_type == "3f":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR5/ur5_3f.usd"

        # robot
        self._robots = []
        self._robot_position = np.array([0.269, 0.1778, 0.0])  # Gf.Vec3f(0.269, 0.1778, 0.0)
        self._robot_orientation = np.array([1, 0, 0, 0])
        return

    def add_light(self):
        # Create a sphere light in Industrial unit
        sphereLight = UsdLux.SphereLight.Define(self._stage, Sdf.Path(f"/World/Env_{self._index}/sphereLight_{self._index}"))
        sphereLight.CreateIntensityAttr(100000)
        translate = (0.0, 0.0, 7.0)
        tx, ty, tz = tuple(translate)
        sphereLight.AddRotateXYZOp().Set((0, 0, 0))
        sphereLight.AddTranslateOp().Set(Gf.Vec3f(tx, ty, tz))

    def add_robot(self):

        robot = None

        if self._robot_type == "2f85":
            robot = self._world.scene.add(
                UR52F(
                    prim_path=f"/World/Env_{self._index}/robot_{self._index}",
                    usd_path=self.ROBOT_PATH,
                    name=f"robot_{self._index}",
                    # Table과의 충돌을 막기 위해 manual position 지정
                    position=np.array([-0.7, 0.0, 0.0])+self._xyz_offset, 
                    orientation=self._robot_orientation,
                    attach_gripper=True,
                )
            )
        elif self._robot_type == "3f":
            robot = self._world.scene.add(
                UR53F(
                    prim_path=f"/World/Env_{self._index}/robot_{self._index}",
                    usd_path=self.ROBOT_PATH,
                    name=f"robot_{self._index}",
                    position=np.array([-0.7, 0.0, 0.0])+self._xyz_offset, 
                    orientation=self._robot_orientation,
                    attach_gripper=True,
                )
            )
        return

    def _setup_simulation(self):
        super()._setup_simulation()
        # self._scene.enable_gpu_dynamics(flag=False)

    def add_objects(self):
        self.add_table_assets()
        self.add_nuts_bolts_assets()

    def setup_dataset(self):
        self._robo_data_collector = RoboDataCollector(file_name=f"ur5_2f_screw_env{self._index}_" + self._now_str)
        self._save_count = 0
        self._robo_data_collectors.append(self._robo_data_collector)
        return

    def setup_scene(self):
        self._setup_simulation()
        self._stage = omni.usd.get_context().get_stage()
        self._world.scene.add(XFormPrim(prim_path="/World/collisionGroups", name="collision_groups_xform"))

        # TODO: ROS 2

        # TODO: Add BG
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

            # if self._add_bg:
            #     self.add_bg()
            # if self._add_rep:
            #     self.add_camera()
            #     self.add_bg_repliactor()
            #     self.add_manipulator_replicator()
            #     self.setup_replicator()
            if self._collect_robo_data:
                self.setup_dataset()

            if self._verbose:
                print(f"[MultiRoboticRack] Env_{self._index} Loaded")

        return

    def setup_robots(self):
        robot_position = self._robot_position + self._xyz_offset
        robot = self._world.scene.get_object(f"robot_{self._index}")

        robot_pos = np.array(robot_position)
        robot_pos[2] = robot_pos[2] + self._table_height
        robot.set_world_pose(position=robot_pos)
        robot.set_default_state(position=robot_pos)
        robot.gripper.open()

        self._robotHandIncludeRel.AddTarget(robot.prim_path + "/_f85_basic/left_inner_finger_pad")
        self._robotHandIncludeRel.AddTarget(robot.prim_path + "/_f85_basic/right_inner_finger_pad")

        controller = NutBoltController(
            name="nut_bolt_controller", 
            robot=robot,
            robot_type="ur5",
        )

        self._controllers.append(controller)
        self._robots.append(robot)

    def add_objects_physical_properties(self):
        self.add_table_property()
        self.add_nuts_and_bolt_property()

    def create_dataset(self):
        robot = self._world.scene.get_object(f"robot_{self._index}")
        if self._verbose:
            print(f"[create_dataset] self._index : {self._index}")
        robo_data_collector = self._robo_data_collectors[self._index]
        joint_names = robot._articulation_view.joint_names
        robo_data_collector.create_dataset(
            "joint_names",
            {
                "universal_robot_5": joint_names,
            }
        )

    async def setup_post_load(self):
        self._world.scene.enable_bounding_boxes_computations()
        self._setup_materials()

        for i in range(self._num_lens):
            self._index = i
            self._xyz_offset = np.array(self._env_offsets[self._index])

            if self._add_bg:
                self.add_bg_property()
            if self._collect_robo_data:
                self.create_dataset()
            self.add_objects_physical_properties()
            self.setup_robots()

        await self._world.reset_async()
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
    
    def forward(self):

        for i, robot in enumerate(self._robots):

            if self._controllers[i].is_paused():
                self._controllers[i].reset(robot)

            if self._controllers[i]._i < min(self._num_nuts, self._num_bolts):
                nut_geom = self._world.scene.get_object(f"nut0_{i}")
                bolt_geom = self._world.scene.get_object(f"bolt0_{i}")

                nut_position, _ = nut_geom.get_world_pose()
                bolt_position, _ = bolt_geom.get_world_pose()

                initial_position = nut_position
                placing_position = bolt_position + self._top_of_bolt

                self._controllers[i].forward(
                    initial_picking_position=initial_position,
                    bolt_top=placing_position,
                    gripper_to_nut_offset=self._gripper_to_nut_offset,
                    x_offset=0.0,
                )

            if self._collect_robo_data:
                self._robo_data_collectors[i].create_single_pp_data(
                    name=f"step_{self._save_count}",
                    robot=robot,
                    controller=self._controllers[i],
                )

        self._save_count += 1
