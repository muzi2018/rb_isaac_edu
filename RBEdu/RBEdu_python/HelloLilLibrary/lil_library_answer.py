# Copyright 2024 Road Balance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.examples.base_sample import BaseSample
from pxr import UsdGeom, Gf, UsdLux, UsdPhysics
from omni.physx.scripts import physicsUtils

import omni.replicator.core as rep
import numpy as np
import omni.usd
import random


class HelloLilLibrary(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.Library_URL = "omniverse://localhost/Projects/RoadBalanceEdu/Library.usdz"
        self.Office_Computer_URL = "omniverse://localhost/Projects/RoadBalanceEdu/Office_Computer.usdz"
        self.Office_Table_URL = "omniverse://localhost/Projects/RoadBalanceEdu/Office_Table.usdz"
        self.Office_Chair_URL = "omniverse://localhost/Projects/RoadBalanceEdu/Office_chair.usdz"
        self.Wood_Bookcase_URL = "omniverse://localhost/Projects/RoadBalanceEdu/Wood_Bookcase.usdz"
        return
    
    def add_lights(self):
        # Create a Sphere light
        sphereLight1 = UsdLux.SphereLight.Define(self._stage, "/World/SphereLight1")
        sphereLight1.CreateRadiusAttr(0.3)
        sphereLight1.CreateIntensityAttr(200000.0)
        sphereLight1.AddTranslateOp().Set(Gf.Vec3f(-0.8, 9.15454, 4.5))

        sphereLight2 = UsdLux.SphereLight.Define(self._stage, "/World/SphereLight2")
        sphereLight2.CreateRadiusAttr(0.3)
        sphereLight2.CreateIntensityAttr(200000.0)
        sphereLight2.AddTranslateOp().Set(Gf.Vec3f(-0.8, 5.0, 4.5))

        sphereLight3 = UsdLux.SphereLight.Define(self._stage, "/World/SphereLight3")
        sphereLight3.CreateRadiusAttr(0.3)
        sphereLight3.CreateIntensityAttr(200000.0)
        sphereLight3.AddTranslateOp().Set(Gf.Vec3f(-0.8, -8.74915, 4.5))

        sphereLight4 = UsdLux.SphereLight.Define(self._stage, "/World/SphereLight4")
        sphereLight4.CreateRadiusAttr(0.3)
        sphereLight4.CreateIntensityAttr(200000.0)
        sphereLight4.AddTranslateOp().Set(Gf.Vec3f(-0.8, -4.5, 4.5))

    def add_background(self):
        add_reference_to_stage(usd_path=self.Library_URL, prim_path=f"/World/Library")

        library_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Library")
        physicsUtils.set_or_add_translate_op(library_mesh, translate=Gf.Vec3f(0.0, 14.0, 0.0))
        physicsUtils.set_or_add_orient_op(library_mesh, orient=Gf.Quatf(0.7071068, 0.7071068, 0, 0))
        physicsUtils.set_or_add_scale_op(library_mesh, scale=Gf.Vec3f(0.02, 0.02, 0.02))

        # add collision
        library_geom = GeometryPrim(prim_path="/World/Library", name="library_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

    def add_bookshelves(self):
        # bookcase1
        add_reference_to_stage(usd_path=self.Wood_Bookcase_URL, prim_path=f"/World/Bookcase1")
        bookcase1_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Bookcase1")
        # modify pose
        physicsUtils.set_or_add_translate_op(bookcase1_mesh, translate=Gf.Vec3f(
            1.3, 0.8, 0.0
        ))
        physicsUtils.set_or_add_scale_op(bookcase1_mesh, scale=Gf.Vec3f(
            0.01, 0.01, 0.01
        ))
        physicsUtils.set_or_add_orient_op(bookcase1_mesh, orient=Gf.Quatf(
            0.5, 0.5, -0.5, -0.5
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/Bookcase1"))
        library_geom = GeometryPrim(prim_path="/World/Bookcase1", name="bookcase1_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")
        
        # bookcase2
        add_reference_to_stage(usd_path=self.Wood_Bookcase_URL, prim_path=f"/World/Bookcase2")
        bookcase2_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Bookcase2")
        # modify pose
        physicsUtils.set_or_add_translate_op(bookcase2_mesh, translate=Gf.Vec3f(
            1.3, 1.8, 0.0
        ))
        physicsUtils.set_or_add_scale_op(bookcase2_mesh, scale=Gf.Vec3f(
            0.01, 0.01, 0.01
        ))
        physicsUtils.set_or_add_orient_op(bookcase2_mesh, orient=Gf.Quatf(
            0.5, 0.5, -0.5, -0.5
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/Bookcase2"))
        library_geom = GeometryPrim(prim_path="/World/Bookcase2", name="bookcase2_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # bookcase3
        add_reference_to_stage(usd_path=self.Wood_Bookcase_URL, prim_path=f"/World/Bookcase3")
        bookcase3_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Bookcase3")
        # modify pose
        physicsUtils.set_or_add_translate_op(bookcase3_mesh, translate=Gf.Vec3f(
            1.3, 2.8, 0.0
        ))
        physicsUtils.set_or_add_scale_op(bookcase3_mesh, scale=Gf.Vec3f(
            0.01, 0.01, 0.01
        ))
        physicsUtils.set_or_add_orient_op(bookcase3_mesh, orient=Gf.Quatf(
            0.5, 0.5, -0.5, -0.5
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/Bookcase3"))
        library_geom = GeometryPrim(prim_path="/World/Bookcase3", name="bookcase3_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

    def add_computer_desks(self):
        # Office Computer 1
        add_reference_to_stage(usd_path=self.Office_Computer_URL, prim_path=f"/World/OfficeComputer1")
        computer1_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeComputer1")
        # modify pose
        physicsUtils.set_or_add_translate_op(computer1_mesh, translate=Gf.Vec3f(
            -0.2, 1.0, 0.1
        ))
        physicsUtils.set_or_add_scale_op(computer1_mesh, scale=Gf.Vec3f(
            0.003, 0.003, 0.003
        ))
        physicsUtils.set_or_add_orient_op(computer1_mesh, orient=Gf.Quatf(
            0.7071068, 0.7071068, 0, 0
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeComputer1"))
        library_geom = GeometryPrim(prim_path="/World/OfficeComputer1", name="computer1_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Computer 2
        add_reference_to_stage(usd_path=self.Office_Computer_URL, prim_path=f"/World/OfficeComputer2")
        computer2_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeComputer2")
        # modify pose
        physicsUtils.set_or_add_translate_op(computer2_mesh, translate=Gf.Vec3f(
            -0.2, 3.0, 0.1
        ))
        physicsUtils.set_or_add_scale_op(computer2_mesh, scale=Gf.Vec3f(
            0.003, 0.003, 0.003
        ))
        physicsUtils.set_or_add_orient_op(computer2_mesh, orient=Gf.Quatf(
            0.7071068, 0.7071068, 0, 0
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeComputer2"))
        library_geom = GeometryPrim(prim_path="/World/OfficeComputer2", name="computer2_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

    def add_office_tables(self):
        # Office Table 1
        add_reference_to_stage(usd_path=self.Office_Table_URL, prim_path=f"/World/OfficeTable1")
        table1_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeTable1")
        # modify pose
        physicsUtils.set_or_add_translate_op(table1_mesh, translate=Gf.Vec3f(
            -2.5, -0.8, 0.0
        ))
        physicsUtils.set_or_add_scale_op(table1_mesh, scale=Gf.Vec3f(
            0.007, 0.007, 0.007
        ))
        physicsUtils.set_or_add_orient_op(table1_mesh, orient=Gf.Quatf(
            0, 0, 0.7071068, 0.7071068
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeTable1"))
        library_geom = GeometryPrim(prim_path="/World/OfficeTable1", name="table1_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Table 2
        add_reference_to_stage(usd_path=self.Office_Table_URL, prim_path=f"/World/OfficeTable2")
        table2_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeTable2")
        # modify pose
        physicsUtils.set_or_add_translate_op(table2_mesh, translate=Gf.Vec3f(
            -2.5, 0.2, 0.0
        ))
        physicsUtils.set_or_add_scale_op(table2_mesh, scale=Gf.Vec3f(
            0.007, 0.007, 0.007
        ))
        physicsUtils.set_or_add_orient_op(table2_mesh, orient=Gf.Quatf(
            0, 0, 0.7071068, 0.7071068
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeTable2"))
        library_geom = GeometryPrim(prim_path="/World/OfficeTable2", name="table2_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Table 3
        add_reference_to_stage(usd_path=self.Office_Table_URL, prim_path=f"/World/OfficeTable3")
        table3_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeTable3")
        # modify pose
        physicsUtils.set_or_add_translate_op(table3_mesh, translate=Gf.Vec3f(
            -2.5, 1.2, 0.0
        ))
        physicsUtils.set_or_add_scale_op(table3_mesh, scale=Gf.Vec3f(
            0.007, 0.007, 0.007
        ))
        physicsUtils.set_or_add_orient_op(table3_mesh, orient=Gf.Quatf(
            0, 0, 0.7071068, 0.7071068
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeTable3"))
        library_geom = GeometryPrim(prim_path="/World/OfficeTable3", name="table3_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Table 4
        add_reference_to_stage(usd_path=self.Office_Table_URL, prim_path=f"/World/OfficeTable4")
        table4_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeTable4")
        # modify pose
        physicsUtils.set_or_add_translate_op(table4_mesh, translate=Gf.Vec3f(
            -2.5, 2.2, 0.0
        ))
        physicsUtils.set_or_add_scale_op(table4_mesh, scale=Gf.Vec3f(
            0.007, 0.007, 0.007
        ))
        physicsUtils.set_or_add_orient_op(table4_mesh, orient=Gf.Quatf(
            0, 0, 0.7071068, 0.7071068
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeTable4"))
        library_geom = GeometryPrim(prim_path="/World/OfficeTable4", name="table4_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")
        return

    def add_office_chairs(self):
        # Office Chair 1
        add_reference_to_stage(usd_path=self.Office_Chair_URL, prim_path=f"/World/OfficeChair1")
        chair1_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeChair1")
        # modify pose
        physicsUtils.set_or_add_translate_op(chair1_mesh, translate=Gf.Vec3f(
            -2.4, -0.7, 0.0
        ))
        physicsUtils.set_or_add_scale_op(chair1_mesh, scale=Gf.Vec3f(
            0.003, 0.003, 0.003
        ))
        physicsUtils.set_or_add_orient_op(chair1_mesh, orient=Gf.Quatf(
            0.7071068, 0.7071068, 0, 0
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeChair1"))
        library_geom = GeometryPrim(prim_path="/World/OfficeChair1", name="table3_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Chair 1
        add_reference_to_stage(usd_path=self.Office_Chair_URL, prim_path=f"/World/OfficeChair2")
        chair2_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeChair2")
        # modify pose
        physicsUtils.set_or_add_translate_op(chair2_mesh, translate=Gf.Vec3f(
            -2.4, 0.3, 0.0
        ))
        physicsUtils.set_or_add_scale_op(chair2_mesh, scale=Gf.Vec3f(
            0.003, 0.003, 0.003
        ))
        physicsUtils.set_or_add_orient_op(chair2_mesh, orient=Gf.Quatf(
            0.7071068, 0.7071068, 0, 0
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeChair2"))
        library_geom = GeometryPrim(prim_path="/World/OfficeChair2", name="table3_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Chair 1
        add_reference_to_stage(usd_path=self.Office_Chair_URL, prim_path=f"/World/OfficeChair3")
        chair3_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeChair3")
        # modify pose
        physicsUtils.set_or_add_translate_op(chair3_mesh, translate=Gf.Vec3f(
            -2.4, 1.3, 0.0
        ))
        physicsUtils.set_or_add_scale_op(chair3_mesh, scale=Gf.Vec3f(
            0.003, 0.003, 0.003
        ))
        physicsUtils.set_or_add_orient_op(chair3_mesh, orient=Gf.Quatf(
            0.7071068, 0.7071068, 0, 0
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeChair3"))
        library_geom = GeometryPrim(prim_path="/World/OfficeChair3", name="table3_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")

        # Office Chair 4
        add_reference_to_stage(usd_path=self.Office_Chair_URL, prim_path=f"/World/OfficeChair4")
        chair4_mesh = UsdGeom.Mesh.Get(self._stage, "/World/OfficeChair4")
        # modify pose
        physicsUtils.set_or_add_translate_op(chair4_mesh, translate=Gf.Vec3f(
            -2.4, 2.3, 0.0
        ))
        physicsUtils.set_or_add_scale_op(chair4_mesh, scale=Gf.Vec3f(
            0.003, 0.003, 0.003
        ))
        physicsUtils.set_or_add_orient_op(chair4_mesh, orient=Gf.Quatf(
            0.7071068, 0.7071068, 0, 0
        ))
        # add collision & rigid body
        UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath("/World/OfficeChair4"))
        library_geom = GeometryPrim(prim_path="/World/OfficeChair4", name="table3_ref_geom", collision=True)
        library_geom.set_collision_approximation("convexDecomposition")
        return

    def setup_scene(self):
        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()

        self.add_background()
        self.add_lights()
        self.add_bookshelves()
        self.add_computer_desks()
        self.add_office_tables()
        self.add_office_chairs()
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        return

    def setup_scene(self):
        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()

        self.add_background()
        self.add_lights()
        self.add_bookshelves()
        self.add_computer_desks()
        self.add_office_tables()
        self.add_office_chairs()
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        return