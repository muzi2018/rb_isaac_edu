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
from omni.physx.scripts import physicsUtils, particleUtils
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from pxr import UsdGeom, Gf

import numpy as np
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

        self._isaac_assets_path = get_assets_root_path()   # omniverse://localhost/NVIDIA/Assets/Isaac/4.0
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        self.zipper_bag_PATH = self._server_root + "/Projects/RoadBalanceEdu/Blue_Bag.usdz"
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

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self.setup_simulation() # Use GPU
        self.add_robot()

        self._stage = omni.usd.get_context().get_stage()

        zipper_bag_default_prim_path = f"/World/Blue_Bag"
        add_reference_to_stage(usd_path=self.zipper_bag_PATH, prim_path=zipper_bag_default_prim_path)
        zipper_bag_mesh = UsdGeom.Mesh.Get(self._stage, zipper_bag_default_prim_path)

        self.zipper_bag_prim_path_1 = f'/World/Blue_Bag/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/_0/Object_4/Object_0'
        zipper_bag_prim_mesh = UsdGeom.Mesh.Get(self._stage, self.zipper_bag_prim_path_1)
        physicsUtils.set_or_add_translate_op(zipper_bag_mesh, translate=Gf.Vec3f(1.0, -0.19, 0.35))
        physicsUtils.set_or_add_orient_op(zipper_bag_mesh, orient=Gf.Quatf( 0, 1, 0, 0  ))
        physicsUtils.set_or_add_scale_op(zipper_bag_mesh, scale=Gf.Vec3f(0.0025, 0.0025, 0.0025))

        # configure and create particle system
        # so the simulation determines the other offsets from the particle contact offset
        default_prim_path = self._stage.GetPrimAtPath("/World").GetPath()
        
        particle_system_path = default_prim_path.AppendChild("particleSystem")
        particle_system_path = get_default_particle_system_path(self._stage)

        # size rest offset according to plane resolution and width so that particles are just touching at rest
        radius = 0.00365
        particle_contact_offset = radius
        contactOffset = particle_contact_offset
        restOffset = contactOffset * 0.99
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
            solver_position_iterations=64,
        )
        
        # create material and assign it to the system:
        particle_material_path = default_prim_path.AppendChild("particleMaterial")
        particleUtils.add_pbd_particle_material(
            self._stage, 
            particle_material_path, 
            drag=0.1, 
            lift=0.3, 
            friction=3.5,
            adhesion=0.0,
            particle_adhesion_scale=1.0,
            adhesion_offset_scale=0.0,
            particle_friction_scale=5.0,
            vorticity_confinement=0.0,
            cfl_coefficient=1.0,
            gravity_scale=1.0
            )
        physicsUtils.add_physics_material_to_prim(
            self._stage, self._stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # configure as bag
        stretchStiffness = 10000.0
        bendStiffness = 195.0
        shearStiffness = 95.0
        damping = 0.2

        particleUtils.add_physx_particle_cloth(
            stage=self._stage,
            path=self.zipper_bag_prim_path_1,
            dynamic_mesh_path=None,
            particle_system_path=particle_system_path,
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._cube = self._world.scene.get_object("fancy_cube")

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

        plastic_bag_position = np.array([1.0, -0.2, 0.068])
        start_position = plastic_bag_position + np.array([0.0, 0.0, -0.045])
        goal_position = np.array([0.75, 0.0029, 0.15])

        joints_state = self._franka.get_joints_state()
        actions = self._controller.forward(
            picking_position=np.array(start_position),
            placing_position=np.array(goal_position),
            end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi*3/4, 0])),
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
