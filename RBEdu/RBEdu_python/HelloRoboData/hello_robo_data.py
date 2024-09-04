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

from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka

import numpy as np
import datetime
import carb
import h5py
import os

now = datetime.datetime.now()

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


class HelloRoboData(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._sim_count = 0

        now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        self._file_path = current_dir + f"/robo_data_{now_str}.hdf"

        self._sim_count_log = []
        self._cube_position_log = []
        self._cube_rotation_log = []
        self._joint_positions_log = []
        self._joint_velocities_log = []
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self._cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0.3, 0.3, 0.3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 0, 1.0])
            )
        )

        self._franka = world.scene.add(
            Franka(
                prim_path="/World/Fancy_Franka",
                name="fancy_franka"
            )
        )

        self._f = h5py.File(self._file_path, 'w')
        self._group_f = self._f.create_group("franka_pp_data")
        return


    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
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
        cube_position, cube_rotation = self._cube.get_world_pose()

        action = self._controller.forward(
            # picking_position=cube_position,
            picking_position=np.array([0.3, 0.3, 0.02]),
            placing_position=np.array([0.3, -0.3, 0.02]),
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(action)

        joint_positions = self._franka.get_joint_positions()
        joint_velocities = self._franka.get_joint_velocities()

        carb.log_info(f"sim_count: {self._sim_count}")
        carb.log_info(f"cube_position: {cube_position}")
        carb.log_info(f"cube_rotation: {cube_rotation}")
        carb.log_info(f"joint_positions: {joint_positions}")
        carb.log_info(f"joint_velocities: {joint_velocities}")

        self._sim_count += 1

        self._sim_count_log.append(self._sim_count)
        self._joint_positions_log.append(joint_positions)
        self._joint_velocities_log.append(joint_velocities)
        self._cube_position_log.append(cube_position)
        self._cube_rotation_log.append(cube_rotation)

        if self._f is not None:
            print(f"self._sim_count: {self._sim_count}")
            cur_group = self._group_f.create_group(f"step_{self._sim_count}")
            cur_group.attrs["joint_positions"] = joint_positions
            cur_group.attrs["cube_positions"] = cube_position
            cur_group.attrs["cube_rotations"] = cube_rotation
        if self._sim_count == 200:
            self._world.pause()

        if self._controller.is_done():
            self._world.pause()
        return


    def world_cleanup(self):
        try:
            if self._f is not None:
                self._f.close()
                print("Data saved")
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

