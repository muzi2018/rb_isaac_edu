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

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.physx.scripts import deformableUtils, physicsUtils  
from omni.isaac.examples.base_sample import BaseSample

from pxr import UsdGeom, Gf

import omni.physx
import omni.usd
import omni


class HelloDeformable(BaseSample):  
    def __init__(self) -> None: 
        super().__init__()
        return

    def _setup_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_solver_type("TGS")
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)

    def add_red_cube(self):
        # Create cube
        result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        omni.kit.commands.execute("MovePrim", path_from=path, path_to="/World/RedCube")
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/RedCube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.0, 0.5, 0.5))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(0.1, 0.1, 0.1))
        cube_mesh.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
        deformableUtils.add_physx_deformable_body(
            self._stage,
            "/World/RedCube",
            collision_simplification=True,
            simulation_hexahedral_resolution=10,
            self_collision=False,
        )

    def add_blue_cube(self):
        # Create cube
        result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        omni.kit.commands.execute("MovePrim", path_from=path, path_to="/World/BlueCube")
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/BlueCube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.5))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(0.1, 0.1, 0.1))
        cube_mesh.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])
        
        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
        deformableUtils.add_physx_deformable_body(
            self._stage,
            "/World/BlueCube",
            collision_simplification=True,
            simulation_hexahedral_resolution=5,
            self_collision=False,
        )
    
    def add_green_cube(self):
        # Create cube
        result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        omni.kit.commands.execute("MovePrim", path_from=path, path_to="/World/GreenCube")
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/GreenCube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.0, -0.5, 0.5))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(0.1, 0.1, 0.1))
        cube_mesh.CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
        
        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
        deformableUtils.add_physx_deformable_body(
            self._stage,
            "/World/GreenCube",
            collision_simplification=True,
            simulation_hexahedral_resolution=2,
            self_collision=False,
        )

    def setup_scene(self):
        world = self.get_world()

        # Deformable Body feature is only supported on GPU.
        self._setup_simulation()
        
        self._stage = omni.usd.get_context().get_stage()
        world.scene.add_default_ground_plane()
        
        self.add_red_cube()
        self.add_blue_cube()
        self.add_green_cube()

    async def setup_post_load(self):        
        self._world = self.get_world()
        return