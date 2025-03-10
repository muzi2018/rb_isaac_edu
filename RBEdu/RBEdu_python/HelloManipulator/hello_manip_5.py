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

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.examples.base_sample import BaseSample
from omni.physx.scripts import physicsUtils  
from omni.isaac.franka import Franka

from pxr import UsdGeom, Gf, UsdPhysics

import numpy as np
import datetime
import omni
import carb
import h5py
import os

now = datetime.datetime.now()

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)

#### Example 5 - Data collection
class HelloManip(BaseSample):
    
    def __init__(self) -> None:
        super().__init__()
        self._sim_count = 0
        self._isaac_assets_path = get_assets_root_path()
        self.CUBE_URL = self._isaac_assets_path + "/Isaac/Props/Blocks/nvidia_cube.usd"
        
        now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        self._file_path = current_dir + f"/robo_data_{now_str}.hdf"
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
        self._cube = add_reference_to_stage(usd_path=self.CUBE_URL, prim_path=f"/World/NVIDIA_Cube")
        self._stage = omni.usd.get_context().get_stage()

        # modify cube mesh
        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/NVIDIA_Cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.3, -0.3, 0.1))

        # modify cube mass
        cube_mass = UsdPhysics.MassAPI.Apply(get_current_stage().GetPrimAtPath("/World/NVIDIA_Cube"))
        cube_mass.CreateMassAttr().Set(0.1)
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self.add_robot()
        self.add_cube()

        self._f = h5py.File(self._file_path, 'w')
        self._group_f = self._f.create_group("franka_pp_data")
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation  = self._franka
        )
        
        self._franka.gripper.open()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return


    def physics_step(self, step_size):

        current_joint_positions = self._franka.get_joint_positions()

        action = self._controller.forward(
            picking_position=np.array([0.3, -0.3, 0.03]),
            placing_position=np.array([0.3, 0.3, 0.05]),
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(action)

        joint_positions = self._franka.get_joint_positions()
        joint_velocities = self._franka.get_joint_velocities()

        self._sim_count += 1

        if self._f is not None:
            print(f"self._sim_count: {self._sim_count}")
            cur_group = self._group_f.create_group(f"step_{self._sim_count}")
            cur_group.attrs["joint_positions"] = joint_positions
            cur_group.attrs["joint_velocities"] = joint_velocities
        if self._sim_count == 200:
            self._world.pause()

        if self._controller.is_done():
            self._world.pause()
        return

    def world_cleanup(self):
        try:
            if self._f is not None:
                self._f.close()
                print(f"Data saved at {self._file_path}")
            elif self._f is None:
                print("Invalid Operation Data not saved")
        except Exception as e:
            print(e)
        finally:
            self._f = None
        self._save_count = 0

    async def setup_pre_reset(self):
        if self._f is not None:
            self._f.close()
            self._f = None
        elif self._f is None:
            print("Create new file for new data collection...")
            self.setup_dataset()

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

