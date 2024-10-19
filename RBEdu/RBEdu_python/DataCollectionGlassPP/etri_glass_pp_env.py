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
from omni.isaac.etri_env import MultiPickandPlaceEnv
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


class DataCollectionGlassPP(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # franka normal glass cup 
        # self._env_json = current_dir + "/franka_normal_glass_cup.json"
        
        # franka 2f glass cup
        self._env_json = current_dir + "/franka_2f_glass_cup.json"
        
        # franka 3f glass cup
        # self._env_json = current_dir + "/franka_3f_glass_cup.json"
        
        return


    def setup_scene(self):
        self._world = self.get_world()
        
        self.lab_example = MultiPickandPlaceEnv(
            world=self._world,
            json_file_path=self._env_json,
            verbose=True,
        )

        self.lab_example.setup_scene()
        return


    async def setup_post_load(self):
        await self.lab_example.setup_post_load()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return


    def physics_step(self, step_size):
        self.lab_example.forward()
        return


    async def setup_pre_reset(self):
        self.lab_example.setup_pre_reset()
        self._event = 0
        return
    

    def world_cleanup(self):
        self.lab_example.world_cleanup()
        self._world.pause()
        return