import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera

from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import Franka

from PIL import Image
import numpy as np
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


class HelloCamera(BaseSample):

    def __init__(self) -> None:
        super().__init__()
        self._save_count = 0
        self._cube_position = np.array([0.0, 0.5, 0.03])
        return

    def add_camera(self):
        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 5.0]),
            frequency=30,
            resolution=(800, 800),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 90, 0]
            ), degrees=True),
        )
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()
        return

    def add_robot(self):
        self._franka = self._world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="Franka",
            )
        )
        return

    def add_cube(self):
        self.fancy_cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Fancy_cube", 
                name="fancy_cube", 
                position=self._cube_position,
                scale=np.array([0.05, 0.05, 0.05]), 
                color=np.array([0, 0, 1.0]), 
            ))
        return

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()
        
        self.add_camera()
        self.add_robot()
        self.add_cube()

    async def setup_post_load(self):
        self._world = self.get_world()
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._franka.gripper.open()
        self._articulation_controller = self._franka.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback)
        return

    def physics_callback(self, step_size):
        self._camera.get_current_frame()

        joints_state = self._franka.get_joints_state()
        actions = self._controller.forward(
            picking_position=self._cube_position,
            placing_position=np.array([0.5, 0.0, 0.05]),
            current_joint_positions=joints_state.positions,
        )

        if self._controller.is_done():
            self._world.pause()
        else:
            self._articulation_controller.apply_action(actions)

        if self._save_count % 10 == 0:
            rgb_img = self._camera.get_rgb()

            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(rgb_img)

            # Save the image as a PNG file
            file_name = f"{current_dir}/img_{self._save_count}.png"
            image.save(file_name)
            print(f"[Data Collected] {file_name}")

        self._save_count += 1
