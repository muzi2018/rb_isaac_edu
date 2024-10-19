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
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.articulations import Articulation
from omni.isaac.examples.base_sample import BaseSample
from semantics.schema.editor import PrimSemanticData
from omni.physx.scripts import physicsUtils
from pxr import Usd, UsdGeom, Sdf, Gf, Tf

import omni.replicator.core as rep
import numpy as np
import random
import carb
import omni
import os

from omni.isaac.custom_ur10 import UR102F, PickPlaceController

def add_object(
        usd_path: str, 
        prim_path: str,
        translate: tuple = (0.0, 0.0, 0.0),
        orient: tuple = (0.0, 0.0, 0.0, 1.0),
        scale: tuple = (1.0, 1.0, 1.0),
    ) -> None:
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    obj_mesh = UsdGeom.Mesh.Get(get_current_stage(), prim_path)
    tx, ty, tz = tuple(translate)
    physicsUtils.set_or_add_translate_op(
        obj_mesh,
        translate=Gf.Vec3f(tx, ty, tz),
    )
    rw, rx, ry, rz = tuple(orient)
    physicsUtils.set_or_add_orient_op(
        obj_mesh,
        orient=Gf.Quatf(rw, rx, ry, rz),
    )

    # if scale type if tuple
    if isinstance(scale, tuple):
        sx, sy, sz = tuple(scale)
        physicsUtils.set_or_add_scale_op(
            obj_mesh,
            scale=Gf.Vec3f(sx, sy, sz),
        )
    elif isinstance(scale, float):
        physicsUtils.set_or_add_scale_op(
            obj_mesh,
            scale=Gf.Vec3f(scale, scale, scale),
        )

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

        # Nucleus Path Configuration
        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)
        self._root_path = get_assets_root_path()

        self.UR_PATH = self._server_root + "/Projects/ETRI/Manipulator/UR10/ur10_2f85_l515.usd"
        self.Board_URL = "omniverse://localhost/Projects/ETRI/Env/Defection/circuit_board_pin.usd"
        
        self._sim_count = 0
        self._abnormal = False
        self._gripper_opened = False
        return

    def add_ur10(self):
        # Add UR10 robot
        self._ur10 = self._world.scene.add(
            UR102F(
                prim_path="/World/UR10",
                name="UR10",
                usd_path=self.UR_PATH,
                attach_gripper=True,
            )
        )

    def add_circuit_board(self):

        # circuit_board 물체 추가
        prim_path = f"/World/Board"
        add_reference_to_stage(usd_path=self.Board_URL, prim_path=prim_path)
        obj_mesh = UsdGeom.Mesh.Get(get_current_stage(), prim_path)

        add_object(
            usd_path=self.Board_URL,
            prim_path=prim_path,
            translate=(0.513, -0.495, 0.03),
            orient=(0.6963642, 0.6963642, 0.1227878, 0.1227878),
            scale=0.02,
        )

        # Circuit Board은  Articulation 물체이며, 
        # Joint State등 Articulation API를 사용하기 위해 
        # Articulation 객체를 생성하고 초기화 
        self._art_prim = Articulation(
            prim_path=prim_path, 
            name="board"
        )

        # Semantic Data 추가 - 본체와 소켓 핀들에 각기 다른 Semantic Data 추가
        add_semantic_data(prim_path, "circut_board")

        semantic_prim_path = "/World/Board/socket_pin_1"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

        semantic_prim_path = "/World/Board/socket_pin_2"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

        semantic_prim_path = "/World/Board/socket_pin_3"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

        semantic_prim_path = "/World/Board/socket_pin_4"
        add_semantic_data(semantic_prim_path, "socket_pin_normal")

    def setup_scene(self):

        self._world = self.get_world()
        self._world.scene.add_default_ground_plane()

        self.add_circuit_board()
        self.add_ur10()
        return

    async def setup_post_load(self):
        self._robot = self._world.scene.get_object("UR10")
        self._my_gripper = self._robot.gripper

        # Pick and Place Controller
        self._pap_controller = PickPlaceController(
            name="pick_place_controller", 
            gripper=self._robot.gripper, 
            robot_articulation=self._robot,
            fast_movement=False,
            events_dt=[
                0.01,
                0.0035,
                0.01,
                0.25,
                0.008,
                0.005,
                0.005,
                1,
                0.01,
                0.08
            ]
        )

        self._robot.set_joint_positions(
            np.array([0.0, -1.5707, 0.0, 0.0, 0.0, 0.0]), 
            joint_indices=np.array([0, 1, 2, 3, 4, 5])
        )

        # Articulation 초기화 - 반드시 필요
        self._art_prim.initialize()
        # Articulation Object 내에 존재하는 Joint들의 이름을 받아온다.
        self._obj_dof_names = self._art_prim.dof_names

        self._articulation_controller = self._robot.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    def physics_step(self, step_size):

        self._sim_count += 1

        # Articulation Object의 Joint State 조회 (Position, Velocity)
        obj_joint_state = self._art_prim.get_joints_state()
        positions, velocities = obj_joint_state.positions, obj_joint_state.velocities

        # Joint State의 절대값의 합을 계산하여 이 값이 0.1보다 크면 Abnormal로 판단
        pin_joint_sum = np.sum(np.abs(positions))
        if pin_joint_sum > 0.1 and self._abnormal == False:
            self._abnormal = True
            for obj_dof_name in self._obj_dof_names:
                pin_name = obj_dof_name.split('_joint')[0]
                # Semantic Data 수정 - Abnormal로 변경
                modify_semantic_data(
                    prim_path=f"/World/Board/{pin_name}",
                    class_name="socket_pin_abnormal"
                )
            print("Socket Pin Abnormal")

        joints_state = self._robot.get_joints_state()
        actions = self._pap_controller.forward(
            picking_position=np.array([0.5, -0.5, 0.03]),
            placing_position=np.array([0.5, 0.5, 0.03]),
            current_joint_positions=joints_state.positions,
            end_effector_offset=np.array([0.0, 0.0, 0.125]),
            end_effector_orientation=np.array([
                0, 0.7071068, 0, -0.7071068, 
            ]),
        )
        if self._pap_controller.is_done():
            print("done picking and placing")
        else:
            self._articulation_controller.apply_action(actions)
        return

    async def setup_pre_reset(self):
        self._save_count = 0
        self._event = 0
        return

    async def setup_post_reset(self):
        await self._world.play_async()
        return
    
    def world_cleanup(self):
        return