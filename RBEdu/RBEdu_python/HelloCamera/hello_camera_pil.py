import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera

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
        return

    def add_camera(self):
        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 10.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 90, 0]
            ), degrees=True),
        )
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()
        return

    def setup_cube(self, prim_path, name, position, scale, color):
        self._fancy_cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=scale,
                color=color,
            ))
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        
        self.add_camera()
        self.setup_cube(
            prim_path="/World/random_cube",
            name="fancy_cube",
            position=np.array([0, 0, 1.0]),
            scale=np.array([0.5015, 0.5015, 0.5015]),
            color=np.array([1.0, 0.0, 0.0]),
        )

    async def setup_post_load(self):
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback)
        return

    def physics_callback(self, step_size):
        self._camera.get_current_frame()

        if self._save_count % 10 == 0:
            rgb_img = self._camera.get_rgb()

            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(rgb_img)

            # Save the image as a PNG file
            image.save(f"{current_dir}/img_{self._save_count}.png")
            print("Data Collected")

        self._save_count += 1
