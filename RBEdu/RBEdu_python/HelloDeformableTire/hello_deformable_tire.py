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
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts import physicsUtils, deformableUtils
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core import SimulationContext
from pxr import UsdGeom, Gf, PhysxSchema

import omni
import carb


class HelloDeformableTire(BaseSample):

    def __init__(self,world=None) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)
        self.tire_PATH = self._server_root + "/Projects/RoadBalanceEdu/Tire.usdz"
        return

    def add_tire_1(self):   
        deformable_prim_path = f"/World/tire_1"
        add_reference_to_stage(usd_path=self.tire_PATH, prim_path=deformable_prim_path)
        tire_mesh = UsdGeom.Mesh.Define(self._stage, deformable_prim_path)
        self._prim_path_1 = f'/World/tire_1/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/tire_base_Sphere_005_0/Object_4/Object_0'
        physicsUtils.set_or_add_translate_op(tire_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.8))
        physicsUtils.set_or_add_orient_op(tire_mesh, orient=Gf.Quatf( 0.7071068, 0.7071068, 0, 0  ))
        physicsUtils.set_or_add_scale_op(tire_mesh, scale=Gf.Vec3f(0.008, 0.008, 0.008))

    def add_tire_2(self):  
        deformable_prim_path = f"/World/tire_2"

        add_reference_to_stage(usd_path=self.tire_PATH, prim_path=deformable_prim_path)
        tire_mesh = UsdGeom.Mesh.Define(self._stage, deformable_prim_path)

        self._prim_path_2 = f'/World/tire_2/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/tire_base_Sphere_005_0/Object_4/Object_0'
        physicsUtils.set_or_add_translate_op(tire_mesh, translate=Gf.Vec3f(0.0, 1.0, 0.8))
        physicsUtils.set_or_add_orient_op(tire_mesh, orient=Gf.Quatf( 0.7071068, 0.7071068, 0, 0  ))
        physicsUtils.set_or_add_scale_op(tire_mesh, scale=Gf.Vec3f(0.008, 0.008, 0.008))

    def setup_scene(self):
        world = SimulationContext(set_defaults=False)   
        world.scene.add_default_ground_plane()
        physics_context = world.get_physics_context()
        physics_context.enable_gpu_dynamics(True)
        world.set_simulation_dt(physics_dt=1.0 / 600.0, rendering_dt=1.0 / 60.0)

        self._scene = PhysicsContext()
        self._scene.set_solver_type("TGS")
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)

        self._stage = omni.usd.get_context().get_stage()

        self.add_tire_1()
        self.add_tire_2()
        
        return

    def add_tire_1_property(self):
        # Add PhysX deformable body
        deformableUtils.add_physx_deformable_body(
            self._stage,
            self._prim_path_1,
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

        physicsUtils.add_physics_material_to_prim(self._stage, omni.usd.get_prim_at_path(self._prim_path_1), deformable_material_path)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(omni.usd.get_prim_at_path(self._prim_path_1))
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.GetContactOffsetAttr().Set(0.1)

    def add_tire_2_property(self):
        # Add PhysX deformable body
        deformableUtils.add_physx_deformable_body(
            self._stage,
            self._prim_path_2,
            simulation_hexahedral_resolution=18,
            collision_simplification=True,
            self_collision=False,
            solver_position_iteration_count=20,
        )

        def_mat_path = f"/DeformableBodyMaterial_2"
        deformable_material_path = omni.usd.get_stage_next_free_path(self._stage, def_mat_path, True)

        deformableUtils.add_deformable_body_material(
            self._stage,
            deformable_material_path,
            youngs_modulus=100000.0,
            poissons_ratio=0.499,
            damping_scale=0.7,
            elasticity_damping=0.0001,
            dynamic_friction=0.9,
            density=30,
        )

        physicsUtils.add_physics_material_to_prim(self._stage, omni.usd.get_prim_at_path(self._prim_path_2), deformable_material_path)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(omni.usd.get_prim_at_path(self._prim_path_2))
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.GetContactOffsetAttr().Set(0.1)


    async def setup_post_load(self):        
        self._world = self.get_world()

        self.add_tire_1_property()
        self.add_tire_2_property()
        return

