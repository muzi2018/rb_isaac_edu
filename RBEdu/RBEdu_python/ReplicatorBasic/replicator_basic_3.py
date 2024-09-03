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


from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.examples.base_sample import BaseSample
import omni.replicator.core as rep

import datetime
import os

now = datetime.datetime.now()

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


#### Example3 - Spam Can and Light Replicator
class ReplicatorBasic(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()
        self.SPAM_URL = self._isaac_assets_path + "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd"

        now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        self._out_dir = current_dir + f"/rep_output_{now_str}"
        return

    def setup_scene(self):
        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.spam_can = rep.create.from_usd(self.SPAM_URL)

        self.cam1 = rep.create.camera(position=(0, 0, 1.6), look_at=(0, 0, 0))
        self.rp1 = rep.create.render_product(self.cam1, resolution=(640, 480))
        
        self.cam2 = rep.create.camera(position=(1.2, 0, 0.5), look_at=(0, 0, 0.5))
        self.rp2 = rep.create.render_product(self.cam2, resolution=(640, 480))

        self.rp_light = rep.create.light(
            light_type="sphere", 
            temperature=3000,
            intensity=5000.0,
            position=(0.0, 0.0, 1.0), 
            scale=0.5,
            count=1
        )

        rep.randomizer.register(self.random_props)
        rep.randomizer.register(self.random_sphere_lights)
        return

    def random_sphere_lights(self):
        with self.rp_light:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.5, -0.5, 1.0), (0.5, 0.5, 1.0)),
            )

    def random_props(self):
        with self.spam_can:
            rep.modify.semantics([('class', "spam_can")])
            rep.modify.pose(
                position=rep.distribution.uniform((-0.1, -0.1, 0.5), (0.1, 0.1, 0.5)),
                rotation=rep.distribution.uniform((-180,-180, -180), (180, 180, 180)),
                scale = rep.distribution.uniform((0.8), (1.2)),
            )

    async def setup_post_load(self):

        with rep.trigger.on_frame(num_frames=20):
            rep.randomizer.random_props()
            rep.randomizer.random_sphere_lights()

        # Create a writer and apply the augmentations to its corresponding annotators
        writer = rep.WriterRegistry.get("BasicWriter")
        print(f"Writing data to: {self._out_dir}")
        writer.initialize(
            output_dir=self._out_dir, 
            rgb=True,
            bounding_box_2d_tight=True,
            distance_to_camera=True,
            semantic_segmentation=True,
        )

        # Attach render product to writer
        writer.attach([
            self.rp1,
            self.rp2
        ])
        return
