# Copyright 2024 Road Balance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera

from PIL import Image
import numpy as np
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
save_dir = os.path.dirname(current_file_path) + "/images"


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
            resolution=(800, 800),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 90, 0]
            ), degrees=True),
        )
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()
        return

    def add_cube(self, prim_path, name, position, scale, color):
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
        self.add_cube(
            prim_path="/World/random_cube",
            name="fancy_cube",
            position=np.array([0, 0, 1.0]),
            scale=np.array([0.5015, 0.5015, 0.5015]),
            color=np.array([1.0, 0.0, 0.0]),
        )

    async def setup_post_load(self):
        # Check if the folder exists
        if not os.path.exists(save_dir):
            # Create the folder if it doesn't exist
            os.makedirs(save_dir)
            print(f"Folder '{save_dir}' created.")
        else:
            print(f"Folder '{save_dir}' already exists.")

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback)
        return

    def physics_callback(self, step_size):

        self._camera.get_current_frame()

        if self._save_count % 10 == 0:
            rgb_img = self._camera.get_rgb()

            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(rgb_img)

            # Save the image as a PNG file
            save_path = f"{save_dir}/cube_img_{self._save_count}.png"
            image.save(save_path)
            print(f"[Data Collected] {save_path}")

        self._save_count += 1
