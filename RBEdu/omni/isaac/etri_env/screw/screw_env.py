# Nuts Bolt Screw Franka Version

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims.xform_prim import XFormPrim

import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from os.path import expanduser
import numpy as np
import omni.usd
import omni

from omni.isaac.core.physics_context.physics_context import PhysicsContext

from ..multi_env import MultiEnv
from ..multi_env_utils import add_object, add_semantic_data, add_physical_properties


class MultiScrewEnv(MultiEnv):
    def __init__(
            self,
            world=None,
            env_offsets: list = [0.0, 0.0, 0.0],
            add_background: bool = False,
            bg_path: str = None,
            add_replicator: bool = False,
            verbose: bool = False,
            robot_type: str = "2f85", # 2f85, 3f
            add_ros2: bool = False,
            collect_robo_data: bool = False,
        ) -> None:
        
        super().__init__(
            world, 
            env_offsets, 
            add_background, 
            bg_path, 
            add_replicator, 
            verbose, 
            robot_type, 
            add_ros2, 
            collect_robo_data
        )

        self.asset_folder = self._isaac_assets_path + "/Isaac/Samples/Examples/FrankaNutBolt/"

        self.Shop_Table_PATH = self.asset_folder + "SubUSDs/Shop_Table/Shop_Table.usd"
        self.Tooling_Plate_PATH = self.asset_folder + "SubUSDs/Tooling_Plate/Tooling_Plate.usd"
        self.Nut_PATH = self.asset_folder + "SubUSDs/Nut/M20_Nut_Tight_R256_Franka_SI.usd"
        self.Bolt_PATH = self.asset_folder + "SubUSDs/Bolt/M20_Bolt_Tight_R512_Franka_SI.usd"
        self.Pipe_PATH = self.asset_folder + "SubUSDs/Pipe/Pipe.usd"

        # robot
        self._robots = []
        self._robot_position = np.array([0.269, 0.1778, 0.0])  # Gf.Vec3f(0.269, 0.1778, 0.0)
        
        # table and vibra table:
        self._table_height = 0.0
        self._table_position = np.array([0.5, 0.0, 0.0])  # Gf.Vec3f(0.5, 0.0, 0.0)
        self._table_local_position = None # multi env local position
        self._table_scale = 0.01
        self._tooling_plate_offset = np.array([0.0, 0.0, 0.0])

        # TODO nut height 
        self._num_nuts = 2
        self._nut_height = 0.016
        # self._mass_nut = 0.065
        self._mass_nut = 0.025
        self._randomize_nut_positions = True
        self._nut_position_noise_minmax = 0.005
        self._rng_seed = 8

        # pipe and bolt parameters
        # TODO bolt params
        self._num_bolts = 6
        self._bolt_length = 0.1
        self._bolt_radius = 0.11
        self._pipe_pos_on_table = np.array([0.2032, 0.381, 0.0])
        self._bolt_z_offset_to_pipe = 0.08
        self._gripper_to_nut_offset = np.array([0.0, 0.0, 0.003])
        self._top_of_bolt = (
            np.array([0.0, 0.0, self._bolt_length + (self._nut_height / 2)]) + self._gripper_to_nut_offset
        )

        # some global sim options:
        self._solverPositionIterations = 4
        self._solverVelocityIterations = 1
        self._solver_type = "TGS"
        return

    def get_robots(self):
        return self._robots

    def set_world(self, world):
        self._world = world
        return

    def set_env_offsets(self, env_offsets):
        self._env_offsets = env_offsets
        return

    def set_bg_pose(
            self, 
            bg_translation,
            bg_orientation,
            bg_scale,
        ):
        self._bg_translation = bg_translation
        self._bg_orientation = bg_orientation
        self._bg_scale = bg_scale
        return
    
    def add_bg(self):
        prim_path = f"/World/Env_{self._index}/background_{self._index}"
        add_object(
            usd_path=self.BG_PATH,
            prim_path=prim_path,
            scale=self._bg_scale,
            translate=self._bg_translation,
            orient=self._bg_orientation,
        )
        add_semantic_data(prim_path, "background")
        return

    def add_camera(self):
        return

    def add_perspective_repliactor(self):
        return

    def add_manipulator_replicator(self):
        return

    def add_xform(self):
        xform = prims_utils.define_prim(f"/World/Env_{self._index}", prim_type="Xform")
        xformable = UsdGeom.Xformable(xform)
        xformable.AddTranslateOp().Set(value=tuple(self._xyz_offset))
        return

    def add_light(self):
        return

    def add_robot(self):
        from omni.isaac.franka.franka import Franka

        self._world.scene.add(
            Franka(
                prim_path=f"/World/Env_{self._index}/franka_{self._index}",
                name=f"franka_{self._index}"
            )
        )
        return

    def set_replicator_write_path(self, file_name):
        self._out_dir = str(expanduser("~") + "/Documents/ETRI_DATA/" + file_name)
        return

    def setup_replicator(self):
        self._writer = rep.WriterRegistry.get("CustomBasicWriter")
        self._writer.initialize(
            output_dir=self._out_dir, 
            rgb=True,
            bounding_box_2d_tight=True,
            bounding_box_2d_loose=True,
            semantic_segmentation=True,
            instance_id_segmentation=True,
            instance_segmentation=True,
            bounding_box_3d=True,
            distance_to_camera=True,
            pointcloud=True,
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
        )
        return

    def _setup_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        # self._scene.set_solver_type(self._solver_type)
        # self._scene.set_friction_offset_threshold(0.01)
        # self._scene.set_friction_correlation_distance(0.0005)
        # self._scene.set_gpu_total_aggregate_pairs_capacity(10 * 1024)
        # self._scene.set_gpu_found_lost_pairs_capacity(10 * 1024)
        # self._scene.set_gpu_heap_capacity(64 * 1024 * 1024)
        # self._scene.set_gpu_found_lost_aggregate_pairs_capacity(10 * 1024)
        # # added because of new errors regarding collisionstacksize
        # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(get_prim_at_path("/physicsScene"))
        # physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(76000000)  # or whatever min is needed

        # group to include SDF mesh of nut only
        self._meshCollisionGroup = UsdPhysics.CollisionGroup.Define(
            self._world.scene.stage, "/World/collisionGroups/meshColliders"
        )
        collectionAPI = Usd.CollectionAPI.Apply(self._meshCollisionGroup.GetPrim(), "colliders")
        self._nutMeshIncludeRel = collectionAPI.CreateIncludesRel()
        
        # group to include all convex collision (nut convex, pipe, table, vibrating table, other small assets on the table)
        self._convexCollisionGroup = UsdPhysics.CollisionGroup.Define(
            self._world.scene.stage, "/World/collisionGroups/convexColliders"
        )
        collectionAPI = Usd.CollectionAPI.Apply(self._convexCollisionGroup.GetPrim(), "colliders")
        self._convexIncludeRel = collectionAPI.CreateIncludesRel()
        
        # group to include bolt prim only (only has SDF mesh)
        self._boltCollisionGroup = UsdPhysics.CollisionGroup.Define(
            self._world.scene.stage, "/World/collisionGroups/boltColliders"
        )
        collectionAPI = Usd.CollectionAPI.Apply(self._boltCollisionGroup.GetPrim(), "colliders")
        self._boltMeshIncludeRel = collectionAPI.CreateIncludesRel()
        # group to include the robot hands prims only
        self._robotHandCollisionGroup = UsdPhysics.CollisionGroup.Define(
            self._world.scene.stage, "/World/collisionGroups/robotHandColliders"
        )
        collectionAPI = Usd.CollectionAPI.Apply(self._robotHandCollisionGroup.GetPrim(), "colliders")
        self._robotHandIncludeRel = collectionAPI.CreateIncludesRel()

        # invert group logic so only groups that filter each-other will collide:
        self._scene.set_invert_collision_group_filter(True)

        # # the SDF mesh collider nuts should only collide with the bolts
        filteredRel = self._meshCollisionGroup.CreateFilteredGroupsRel()
        filteredRel.AddTarget("/World/collisionGroups/boltColliders")

        # the convex hull nuts should collide with other nuts, 
        # the vibra table, table, pipe and small assets on the table.
        # It should also collide with the robot grippers
        filteredRel = self._convexCollisionGroup.CreateFilteredGroupsRel()
        filteredRel.AddTarget("/World/collisionGroups/convexColliders")
        filteredRel.AddTarget("/World/collisionGroups/robotHandColliders")

        # # the SDF mesh bolt only collides with the SDF mesh nut colliders
        # and with the robot grippers
        filteredRel = self._boltCollisionGroup.CreateFilteredGroupsRel()
        filteredRel.AddTarget("/World/collisionGroups/meshColliders")
        filteredRel.AddTarget("/World/collisionGroups/robotHandColliders")

    def add_table_assets(self):
        prim_path = f"/World/Env_{self._index}/table_{self._index}"
        add_reference_to_stage(usd_path=self.Shop_Table_PATH, prim_path=prim_path)
        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"table_{self._index}", collision=True))

        prim_path = f"/World/Env_{self._index}/tooling_plate_{self._index}"
        add_reference_to_stage(usd_path=self.Tooling_Plate_PATH, prim_path=prim_path)
        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"tooling_plate_{self._index}", collision=True))

        prim_path = f"/World/Env_{self._index}/pipe_{self._index}"
        add_reference_to_stage(usd_path=self.Pipe_PATH, prim_path=prim_path)
        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"pipe_{self._index}", collision=True))
        return

    def add_nuts_bolts_assets(self):
        for bolt in range(self._num_bolts):
            # bolt[object_index]_[Env_index]
            prim_path = f"/World/Env_{self._index}/bolt{bolt}_{self._index}"
            add_reference_to_stage(usd_path=self.Bolt_PATH, prim_path=prim_path)
            self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"bolt{bolt}_{self._index}"))
        for nut in range(self._num_nuts):
            prim_path = f"/World/Env_{self._index}/nut{nut}_{self._index}"
            add_reference_to_stage(usd_path=self.Nut_PATH, prim_path=prim_path)
            self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"nut{nut}_{self._index}"))
        return

    def setup_scene(self):
        self._setup_simulation()
        self._stage = omni.usd.get_context().get_stage()
        self._world.scene.add(XFormPrim(prim_path="/World/collisionGroups", name="collision_groups_xform"))

        for i in range(len(self._env_offsets)):
            self._index = i
            self._xyz_offset = np.array(self._env_offsets[self._index])

            self.add_xform()
            # self.add_camera()
            # self.add_light()
            self.add_table_assets()
            self.add_nuts_bolts_assets()
            self.add_robot()
            if self._add_bg:
                self.add_bg()
            else:
                self._world.scene.add_default_ground_plane()
            # self.add_perspective_repliactor()
            # self.add_manipulator_replicator()

        # self.setup_replicator()
        return

    def _setup_materials(self):
        self._bolt_physics_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/BoltMaterial",
            name="bolt_material_physics",
            static_friction=0.2,
            dynamic_friction=0.2,
        )
        self._nut_physics_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/NutMaterial",
            name="nut_material_physics",
            static_friction=0.2,
            dynamic_friction=0.2,
        )
        self._robot_finger_physics_material = PhysicsMaterial(
            prim_path="/World/PhysicsMaterials/RobotFingerMaterial",
            name="robot_finger_material_physics",
            static_friction=0.7,
            dynamic_friction=0.7,
        )

    # async def add_table_property(self):
    def add_table_property(self):
        ##shop_table
        self._table_local_position = self._table_position + self._xyz_offset
        self._table_ref_geom = self._world.scene.get_object(f"table_{self._index}")
        self._table_ref_geom.set_local_scale(np.array([self._table_scale]))
        self._table_ref_geom.set_world_pose(position=self._table_local_position)
        self._table_ref_geom.set_default_state(position=self._table_local_position)
        lb = self._world.scene.compute_object_AABB(name=f"table_{self._index}")
        zmin = lb[0][2]
        zmax = lb[1][2]
        self._table_local_position[2] = -zmin
        self._table_height = zmax
        self._table_ref_geom.set_collision_approximation("none")
        self._convexIncludeRel.AddTarget(self._table_ref_geom.prim_path)

        ##tooling_plate
        self._tooling_plate_geom = self._world.scene.get_object(f"tooling_plate_{self._index}")
        self._tooling_plate_geom.set_local_scale(np.array([self._table_scale]))
        lb = self._world.scene.compute_object_AABB(name=f"tooling_plate_{self._index}")
        zmin = lb[0][2]
        zmax = lb[1][2]
        tooling_transform = self._tooling_plate_offset
        tooling_transform[2] = -zmin + self._table_height
        tooling_transform = tooling_transform + self._table_local_position
        self._tooling_plate_geom.set_world_pose(position=tooling_transform)
        self._tooling_plate_geom.set_default_state(position=tooling_transform)
        self._tooling_plate_geom.set_collision_approximation("boundingCube")
        self._table_height += zmax - zmin
        self._convexIncludeRel.AddTarget(self._tooling_plate_geom.prim_path)

        ##pipe
        self._pipe_geom = self._world.scene.get_object(f"pipe_{self._index}")
        self._pipe_geom.set_local_scale(np.array([self._table_scale]))
        lb = self._world.scene.compute_object_AABB(name=f"pipe_{self._index}")
        zmin = lb[0][2]
        zmax = lb[1][2]
        self._pipe_height = zmax - zmin
        pipe_transform = self._pipe_pos_on_table
        pipe_transform[2] = -zmin + self._table_height
        pipe_transform = pipe_transform + self._table_local_position
        self._pipe_geom.set_world_pose(position=pipe_transform, orientation=np.array([0, 0, 0, 1]))
        self._pipe_geom.set_default_state(position=pipe_transform, orientation=np.array([0, 0, 0, 1]))
        self._pipe_geom.set_collision_approximation("none")
        self._convexIncludeRel.AddTarget(self._pipe_geom.prim_path)
        # await self._world.reset_async()

    # async def add_nuts_and_bolt_property(self, add_debug_nut=False):
    def add_nuts_and_bolt_property(self, add_debug_nut=False):
        angle_delta = np.pi * 2.0 / self._num_bolts
        for j in range(self._num_bolts):
            self._bolt_geom = self._world.scene.get_object(f"bolt{j}_{self._index}")
            
            # self._bolt_geom.prim.SetInstanceable(True)
            
            bolt_pos = np.array(self._pipe_pos_on_table) + self._table_local_position
            bolt_pos[0] += np.cos(j * angle_delta) * self._bolt_radius
            bolt_pos[1] += np.sin(j * angle_delta) * self._bolt_radius
            bolt_pos[2] = self._bolt_z_offset_to_pipe + self._table_height
            self._bolt_geom.set_world_pose(position=bolt_pos)
            self._bolt_geom.set_default_state(position=bolt_pos)

            print(f"self._bolt_geom.prim_path: {self._bolt_geom.prim_path}")
            self._boltMeshIncludeRel.AddTarget(self._bolt_geom.prim_path)
            # self._bolt_geom.apply_physics_material(self._bolt_physics_material)

        self._nut_init_poses = np.array(
            [
                [0.6, 0.0, 0.85, 1.0, 0.0, 0.0, 0.0],
                # Here
                # [0.81478, 0.38028, 1.02418, 1.0, 0.0, 0.0, 0.0],
                [0.6, -0.1, 0.85, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        for nut_idx in range(self._num_nuts):
            nut_pos = self._nut_init_poses[nut_idx, :3].copy() + self._xyz_offset

            self._nut_geom = self._world.scene.get_object(f"nut{nut_idx}_{self._index}")
            # self._nut_geom.prim.SetInstanceable(True)
            self._nut_geom.set_world_pose(position=np.array(nut_pos.tolist()))
            self._nut_geom.set_default_state(position=np.array(nut_pos.tolist()))
            
            # physxRBAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(self._nut_geom.prim)
            
            # physxRBAPI.CreateSolverPositionIterationCountAttr().Set(self._solverPositionIterations)
            # physxRBAPI.CreateSolverVelocityIterationCountAttr().Set(self._solverVelocityIterations)
            
            # self._nut_geom.apply_physics_material(self._nut_physics_material)
            
            print(f"self._nut_geom.prim_path: {self._nut_geom.prim_path}")
            self._convexIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_Convex")
            self._nutMeshIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_SDF")
            
            # rbApi3 = UsdPhysics.RigidBodyAPI.Apply(self._nut_geom.prim.GetPrim())
            
            # rbApi3.CreateRigidBodyEnabledAttr(True)

            physxAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(self._nut_geom.prim.GetPrim())
            physxAPI.CreateSleepThresholdAttr().Set(0.0)
            
            # massAPI = UsdPhysics.MassAPI.Apply(self._nut_geom.prim.GetPrim())
            # massAPI.CreateMassAttr().Set(self._mass_nut)
        # await self._world.reset_async()

    async def setup_robots(self):
        robot_position = self._robot_position + self._xyz_offset
        robot = self._world.scene.get_object(f"franka_{self._index}")
        robot_pos = np.array(robot_position)
        robot_pos[2] = robot_pos[2] + self._table_height
        robot.set_world_pose(position=robot_pos)
        robot.set_default_state(position=robot_pos)
        robot.gripper.open()

        kps = np.array([6000000.0, 600000.0, 6000000.0, 600000.0, 25000.0, 15000.0, 25000.0, 15000.0, 15000.0])
        kds = np.array([600000.0, 60000.0, 300000.0, 30000.0, 3000.0, 3000.0, 3000.0, 6000.0, 6000.0])
        robot.get_articulation_controller().set_gains(kps=kps, kds=kds, save_to_usd=True)
        
        self._robotHandIncludeRel.AddTarget(robot.prim_path + "/panda_leftfinger")
        self._robotHandIncludeRel.AddTarget(robot.prim_path + "/panda_rightfinger")

        prim_path = f"/World/Env_{self._index}/franka_{self._index}"

        robot_left_finger = self._world.stage.GetPrimAtPath(
            prim_path + "/panda_leftfinger/geometry/panda_leftfinger"
        )
        x = UsdShade.MaterialBindingAPI.Apply(robot_left_finger)
        x.Bind(
            self._robot_finger_physics_material.material,
            bindingStrength="weakerThanDescendants",
            materialPurpose="physics",
        )

        robot_right_finger = self._world.stage.GetPrimAtPath(
            prim_path + "/panda_rightfinger/geometry/panda_rightfinger"
        )
        x2 = UsdShade.MaterialBindingAPI.Apply(robot_right_finger)
        x2.Bind(
            self._robot_finger_physics_material.material,
            bindingStrength="weakerThanDescendants",
            materialPurpose="physics",
        )
        self._robots.append(robot)
        await self._world.reset_async()

    async def setup_post_load(self):
        self._world.scene.enable_bounding_boxes_computations()
        await self._setup_materials()

        for i in range(self._num_lens):
            self._index = i
            # screw example은 여기에서 위치가 설정된다.
            self._xyz_offset = np.array(self._env_offsets[self._index])

            if self._add_bg:
                self.add_bg_property()
            if self._collect_robo_data:
                self.create_dataset()
            await self.add_objects_physical_properties()
            await self.setup_robots()
        return

    def add_bg_property(self):
        prim_path = f"/World/Env_{self._index}/background_{self._index}"
        add_physical_properties(
            prim_path=prim_path,
            rigid_body=False,
            collision_approximation="meshSimplification"
        )
        return
    
    def forward(self): 
        return
