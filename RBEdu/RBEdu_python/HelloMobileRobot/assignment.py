from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.examples.base_sample import BaseSample
from omni.physx.scripts import physicsUtils
from omni.isaac.sensor import Camera


import numpy as np
import cv2


class HelloMobileRobot(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._sim_count = 0
        assets_root_path = get_assets_root_path()
        self.JETBOT_PATH = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        # self.TRACK_PATH = 
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_ground_plane()

        # add_reference_to_stage(usd_path=self.TRACK_PATH, prim_path="/World/track")
        
        self._jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/fancy_jetbot",
                name="fancy_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=self.JETBOT_PATH,
                position=np.array([1.5, 0.35, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        self._camera = Camera(
            prim_path="/World/fancy_jetbot/chassis/rgb_camera/jetbot_camera",
            resolution=(640, 480),
        )
        self._camera.initialize()

        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_jetbot")

        self._diff_controller = DifferentialController(
            name="simple_control",
            wheel_radius=0.03,
            wheel_base=0.1125
        )
        self._diff_controller.reset()

        self._pose_controller = WheelBasePoseController(
            name="pose_controller",
            open_loop_wheel_controller=self._diff_controller,
            is_holonomic=False
        )
        self._pose_controller.reset()

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        return

    def send_robot_actions(self, step_size):
        position, orientation = self._jetbot.get_world_pose()
        self._jetbot.apply_action(
            self._pose_controller.forward(
                start_position=position,
                start_orientation=orientation,
                goal_position=np.array([0.0, 0.0])
            )
        )

        self._camera.get_current_frame()
        bgr_img = self._camera.get_rgb()
        rgb_img = bgr_img[..., ::-1]
        cv2.imwrite(f"/home/user/Documents/img_{self._sim_count}.png", rgb_img)


        self._sim_count += 1
        return