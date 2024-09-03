# Copyright 2024 Road Balance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from omni.isaac.examples.base_sample import BaseSample
import omni.replicator.core as rep

import datetime
import os

now = datetime.datetime.now()

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)

#### Example1 - Color Replicator
class ReplicatorBasic(BaseSample):

    def __init__(self) -> None:
        super().__init__()

        now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        self._out_dir = current_dir + f"/rep_output_{now_str}"
        return

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.cube_mat = rep.create.material_omnipbr(
            diffuse=rep.distribution.uniform((0,0,0), (1,1,1)), 
            count=100
        )

        self.red_cube = rep.create.cube(position=(0, 0, 0.71))
        self.cam = rep.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
        self.rp = rep.create.render_product(self.cam, resolution=(1024, 1024))
        
        rep.randomizer.register(self.color_randomize)
        return

    def color_randomize(self):
        with self.red_cube:
            rep.randomizer.materials(self.cube_mat)
        return self.red_cube.node

    async def setup_post_load(self):

        with rep.trigger.on_frame(num_frames=20):
            rep.randomizer.color_randomize()

        # Create a writer and apply the augmentations to its corresponding annotators
        writer = rep.WriterRegistry.get("BasicWriter")
        print(f"Writing data to: {self._out_dir}")
        writer.initialize(
            output_dir=self._out_dir, 
            rgb=True,
            bounding_box_2d_tight=True,
            distance_to_camera=True
        )

        # Attach render product to writer
        writer.attach([self.rp])
        return
