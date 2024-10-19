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
from omni.isaac.core.articulations import Articulation
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.etri_env import MultiPickandPlaceEnv
from semantics.schema.editor import PrimSemanticData
import omni.replicator.core as rep
import numpy as np
import datetime
import os

now = datetime.datetime.now()

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


def add_semantic_data(
        prim_path: str, 
        class_name: str
    ) -> None:
    prim = get_current_stage().GetPrimAtPath(prim_path)
    prim_sd = PrimSemanticData(prim)
    prim_sd.add_entry("class", class_name)

def modify_semantic_data(
        prim_path: str, 
        class_name: str
    ) -> None:
    socket_pin_test = rep.get.prim_at_path(prim_path)
    with socket_pin_test:
        rep.modify.semantics([("class", class_name)])


class DefectEnv(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._abnormal = False
        self._env_json = current_dir + "/example_circuit_board.json"

        return


    def setup_scene(self):
        self._world = self.get_world()
        
        self.lab_example = MultiPickandPlaceEnv(
            world=self._world,
            json_file_path=self._env_json,
            verbose=True,
        )
        self.lab_example.setup_scene()

        self._art_prim = Articulation(
            prim_path="/World/Env_0/Circuit_Board_Pre_Dynamics", 
            name="circuit_board"
        )

        semantic_prim_path = "/World/Env_0/Circuit_Board_Pre_Dynamics/socket_pin_1"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

        semantic_prim_path = "/World/Env_0/Circuit_Board_Pre_Dynamics/socket_pin_2"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

        semantic_prim_path = "/World/Env_0/Circuit_Board_Pre_Dynamics/socket_pin_3"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

        semantic_prim_path = "/World/Env_0/Circuit_Board_Pre_Dynamics/socket_pin_4"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")
        return


    async def setup_post_load(self):
        await self.lab_example.setup_post_load()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        
        self._art_prim.initialize()
        self._obj_dof_names = self._art_prim.dof_names
        return


    def physics_step(self, step_size):
        self.lab_example.forward()

        obj_joint_state = self._art_prim.get_joints_state()
        positions, velocities = obj_joint_state.positions, obj_joint_state.velocities

        pin_joint_sum = np.sum(np.abs(positions))
        print(f"pin_joint_sum: {pin_joint_sum} / self._abnormal: {self._abnormal}")
        if pin_joint_sum > 0.1 and self._abnormal == False:
            self._abnormal = True
            for obj_dof_name in self._obj_dof_names:
                pin_name = obj_dof_name.split('_joint')[0]
                modify_semantic_data(
                    prim_path=f"/World/Env_0/Circuit_Board_Pre_Dynamics/{pin_name}",
                    class_name="socket_pin_abnormal"
                )
            print("Socket Pin Abnormal")
        
        return


    async def setup_pre_reset(self):
        self.lab_example.setup_pre_reset()
        self._event = 0
        return
    

    def world_cleanup(self):
        self.lab_example.world_cleanup()
        self._world.pause()
        return