# Pick and Place UR10 Version
import numpy as np
import omni.usd
import omni

from omni.isaac.custom_controller import CustomPickPlaceController
from omni.isaac.ur5 import UR52F, UR53F

from ..multi_env_utils import add_physical_properties, add_material_to_prim
from ..multi_env_utils import add_object, add_semantic_data

from .pick_and_place_env import MultiRoboticRack


class MultiRoboticRackUR5(MultiRoboticRack):
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
        ) -> None:

        super().__init__(world, env_offsets, add_background, bg_path, add_replicator, verbose, robot_type, add_ros2)

        self._robot_type = robot_type
        if self._robot_type == "2f85":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR5/ur5_2f85.usd"
        elif self._robot_type == "3f":
            self.ROBOT_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR5/ur5_3f.usd"

        self.GLASS_CUP_PATH = self._server_root + "/Projects/ETRI/Glass/Glass_cup.usdz"

        self.BG_PATH = ""
        self._bg_scale = (1, 1, 1)
        self._bg_translation = (0, 0, 0)
        self._bg_orientation = (1, 0, 0, 0)
        
        self._robot_position = np.array([0.4, 0.5, 0.66])
        self._robot_orientation = np.array([0.7071068, 0, 0, -0.7071068])
        return

    def add_robot(self):

        robot = None

        if self._robot_type == "2f85":
            robot = self._world.scene.add(
                UR52F(
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
                UR53F(
                    prim_path=f"/World/Env_{self._index}/robot_{self._index}",
                    usd_path=self.ROBOT_PATH,
                    name=f"robot_{self._index}",
                    position=self._robot_position + self._xyz_offset,
                    orientation=self._robot_orientation,
                    attach_gripper=True,
                )
            )

        controller = CustomPickPlaceController(
            name="pick_place_controller",
            gripper=robot.gripper,
            robot_articulation=robot,
            end_effector_initial_height=1.2,
            robot_type="ur5",
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

    def setup_scene(self):
        self._stage = omni.usd.get_context().get_stage()

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
            # 카메라 추가하면 급격히 느려지므로 우선 주석
            # self.add_camera()

            self.add_light()
            self.add_robot()
            self.add_objects()

            # TODO: 배경도 추가해보자. 아직 구현 안해두었음
            # 배경을 하나의 usd로 만들어서 추가하면 편할 것 같음
            if self._add_bg:
                self.add_bg()
            if self._add_rep:
                self.add_bg_repliactor()
                self.add_manipulator_replicator()

            if self._verbose:
                print(f"[MultiRoboticRack] Env_{self._index} Loaded")

        if self._add_rep:
            self.setup_replicator()
        return

    def add_objects_physical_properties(self):
        self.add_industrial_table_1_property()
        self.add_industrial_steel_table_1_property()
        self.add_cardbox_property()
        self.add_wine_glass_property()

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

    def setup_post_load(self):
        self._world.scene.enable_bounding_boxes_computations()
        self.create_materials()

        for i in range(self._num_lens):
            self._index = i
            if self._add_bg:
                self.add_bg_property()
            
            self.setup_robots()
            self.add_objects_physical_properties()
        return