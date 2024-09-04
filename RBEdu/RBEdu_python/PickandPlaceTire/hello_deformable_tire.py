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


from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.franka.controllers import PickPlaceController
from omni.physx.scripts import physicsUtils, deformableUtils
from omni.isaac.franka import Franka

from pxr import UsdGeom, Gf
import numpy as np
import omni
import carb

from omni.isaac.examples.base_sample import BaseSample


class HelloDeformableTire(BaseSample):

    def __init__(self) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()   # omniverse://localhost/NVIDIA/Assets/Isaac/4.0
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        self.TIRE_PATH = self._server_root + "/Projects/RoadBalanceEdu/Tire.usdz"
        self.Franka_PATH = self._server_root + "Projects/RoadBalanceEdu/franka_realsense.usd"
        
        self._franka_position = np.array([0.7, 0.4, 0.0])
        self._franka_orientation = np.array([0.7071068, 0, 0, -0.7071068])
        return

    def setup_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)

    def add_robot(self):
        self._franka = self._world.scene.add(
            Franka(
                prim_path=f"/World/Franka",
                name=f"Franka",
                position=self._franka_position,
                orientation=self._franka_orientation,
            )
        )
        return

    def add_tire_1_property(self):
        # Add PhysX deformable body
        self._deformable_prim_path = f'/World/tire_1/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/tire_base_Sphere_005_0/Object_4/Object_0'

        deformableUtils.add_physx_deformable_body(
            self._stage,
            self._deformable_prim_path,
            simulation_hexahedral_resolution=18,
            collision_simplification=True,
            self_collision=False,
            solver_position_iteration_count=20,
        )

        def_mat_path = f"/DeformableBodyMaterial_1"
        deformable_material_path = omni.usd.get_stage_next_free_path(self._stage, def_mat_path, True)
        deformableUtils.add_deformable_body_material(
            self._stage,
            deformable_material_path,
            youngs_modulus=100000.0,
            poissons_ratio=0.499,
            damping_scale=0.2,
            elasticity_damping=0.0001,
            dynamic_friction=0.9,
            density=30,
        )

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self.setup_simulation() # Use GPU
        self.add_robot()

        self._stage = omni.usd.get_context().get_stage()

        # Tire_1
        deformable_prim_path = f"/World/tire_1"
        add_reference_to_stage(usd_path=self.TIRE_PATH, prim_path=deformable_prim_path)
        tire_mesh = UsdGeom.Mesh.Define(self._stage, deformable_prim_path)
        physicsUtils.set_or_add_translate_op(tire_mesh, translate=Gf.Vec3f(1.07, -0.15, 0.068))
        physicsUtils.set_or_add_orient_op(tire_mesh, orient=Gf.Quatf( 0.7071068, 0.7071068, 0, 0  ))
        physicsUtils.set_or_add_scale_op(tire_mesh, scale=Gf.Vec3f(0.002, 0.002, 0.002))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._cube = self._world.scene.get_object("fancy_cube")

        self.add_tire_1_property()

        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

        self._franka.gripper.open()
        self._articulation_controller = self._franka.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return

    def physics_step(self, step_size):

        tire_position = np.array([1.0, -0.2, 0.068])
        start_position = tire_position + np.array([0.0, 0.0, -0.045])
        goal_position = np.array([0.75, 0.0029, 0.30])

        joints_state = self._franka.get_joints_state()
        actions = self._controller.forward(
            picking_position=np.array(start_position),
            placing_position=np.array(goal_position),
            end_effector_orientation=euler_angles_to_quat(
                    np.array([0, np.pi*3/4, 0])
                ),
            end_effector_offset=np.array([0, 0, 0.0]),
            current_joint_positions=joints_state.positions,
        )

        if self._controller.is_done():
            self._world.pause()
        else:
            self._articulation_controller.apply_action(actions)
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
