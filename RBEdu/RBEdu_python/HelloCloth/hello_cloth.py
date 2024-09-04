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
from omni.physx.scripts import physicsUtils, particleUtils

from omni.isaac.examples.base_sample import BaseSample
from pxr import UsdGeom, Gf, UsdPhysics
import omni
import carb


def get_default_particle_system_path(stage):
    if not stage.GetDefaultPrim():
        default_prim_xform = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(default_prim_xform.GetPrim())
    return stage.GetDefaultPrim().GetPath().AppendChild("ParticleSystem")


class HelloDeformableCloth(BaseSample):

    def __init__(self) -> None:
        super().__init__()
  
        self._isaac_assets_path = get_assets_root_path()
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        # Plastic Bag Dirty 3
        self.Plastic_Bag_Dirty_3_PATH = self._server_root + "/Projects/RoadBalanceEdu/Plastic_Bag_Dirty_3.usdz"

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        world.set_simulation_dt(physics_dt=1.0 / 600.0, rendering_dt=1.0 / 60.0)   #change dt    #ì‹œìž‘####

        self._scene = PhysicsContext()
        self._scene.set_solver_type("TGS")
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._stage = omni.usd.get_context().get_stage()

        default_prim_path = self._stage.GetPrimAtPath("/World").GetPath()

        Plastic_Bag_Dirty_3_default_prim_path = f"/World/Plastic_Bag_Dirty_3_1"
        add_reference_to_stage(usd_path=self.Plastic_Bag_Dirty_3_PATH, prim_path=Plastic_Bag_Dirty_3_default_prim_path)
        Plastic_Bag_Dirty_3_mesh = UsdGeom.Mesh.Get(self._stage, Plastic_Bag_Dirty_3_default_prim_path)

        self.Plastic_Bag_Dirty_3_prim_path_1 = f'/World/Plastic_Bag_Dirty_3_1/Meshes/Sketchfab_model/Collada_visual_scene_group/PlasticBag3/defaultMaterial/defaultMaterial'
        Plastic_Bag_Dirty_3_prim_mesh = UsdGeom.Mesh.Get(self._stage, self.Plastic_Bag_Dirty_3_prim_path_1)
        physicsUtils.set_or_add_translate_op(Plastic_Bag_Dirty_3_mesh, translate=Gf.Vec3f(0.4, 0, 0.5))
        physicsUtils.set_or_add_orient_op(Plastic_Bag_Dirty_3_mesh, orient=Gf.Quatf( 0, 1, 0, 0  ))
        physicsUtils.set_or_add_scale_op(Plastic_Bag_Dirty_3_mesh, scale=Gf.Vec3f(0.008, 0.008, 0.008))

        particle_system_path = default_prim_path.AppendChild("particleSystem")
        particle_system_path = get_default_particle_system_path(self._stage)

        radius = 0.04
        particle_contact_offset = radius
        contactOffset = particle_contact_offset
        restOffset = radius * 0.99
        solid_rest_offset = radius * 0.99
        fluid_rest_offset = radius * 0.99 * 0.6

        particleUtils.add_physx_particle_system(
            stage=self._stage,
            particle_system_path=particle_system_path,
            contact_offset=contactOffset,
            rest_offset=restOffset,
            particle_contact_offset=particle_contact_offset,
            solid_rest_offset=solid_rest_offset,
            fluid_rest_offset=fluid_rest_offset,
            solver_position_iterations=16,
        )
        
        # create material and assign it to the system:
        particle_material_path = default_prim_path.AppendChild("particleMaterial")
        particleUtils.add_pbd_particle_material(self._stage, particle_material_path)
        particleUtils.add_pbd_particle_material(
            self._stage, 
            particle_material_path, 
            density=0.1,
            friction=0.6,
            damping=0.0,
            cohesion=0.0,
            surface_tension=0.0,
            drag=0.1, 
            lift=0.3,
            adhesion=0.1
        )
        
        physicsUtils.add_physics_material_to_prim(
            self._stage, self._stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # configure as bag
        stretchStiffness = 8000.0
        bendStiffness = 195.0
        shearStiffness = 90.0
        damping = 0.1

        particleUtils.add_physx_particle_cloth(
            stage=self._stage,
            path=self.Plastic_Bag_Dirty_3_prim_path_1,
            dynamic_mesh_path=None,
            particle_system_path=particle_system_path,
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )

        # configure mass:
        particle_mass = 0.2
        num_verts = len(Plastic_Bag_Dirty_3_prim_mesh.GetPointsAttr().Get())
        mass = particle_mass * num_verts
        massApi = UsdPhysics.MassAPI.Apply(Plastic_Bag_Dirty_3_prim_mesh.GetPrim())
        massApi.GetMassAttr().Set(mass)
        return