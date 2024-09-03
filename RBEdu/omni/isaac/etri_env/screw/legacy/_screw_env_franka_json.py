from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.custom_controller import CustomInsertionController
from omni.isaac.custom_controller import NutBoltController 

from ...multi_env_randomizer import MultiEnvRandomizer

import numpy as np
import logging
import carb
import omni

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.xform_prim import XFormPrim

from ..screw_env import MultiScrewEnv


class MultiScrewEnvFrankaJSON(MultiEnvRandomizer):
    def __init__(
            self,
            world,
            json_file_path=None,
            verbose=True,
        ) -> None:

        super().__init__(
            world, 
            json_file_path,
            verbose
        )

        self._num_nuts = 0
        self._num_bolts = 0

        self.Nut_PATH = "omniverse://localhost/Projects/ETRI/Env/BoltNuts/Nut/M20_Nut_Tight_R256_Franka_SI.usd"
        self.Bolt_PATH = "omniverse://localhost/Projects/ETRI/Env/BoltNuts/Bolt/M20_Bolt_Tight_R512_Franka_SI.usd"
        
        self.Shop_Table_PATH = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/Examples/FrankaNutBolt/SubUSDs/Shop_Table/Shop_Table.usd"
        self.Tooling_Plate_PATH = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/Examples/FrankaNutBolt/SubUSDs/Tooling_Plate/Tooling_Plate.usd"
        self.Pipe_PATH = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/Examples/FrankaNutBolt/SubUSDs/Pipe/Pipe.usd"
        return


    def _setup_simulation(self):
        from omni.isaac.core.physics_context.physics_context import PhysicsContext

        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)

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


    def add_table_assets(self, i, env_offset):
        prim_path = f"/World/Env_{i}/table_{i}"
        add_reference_to_stage(usd_path=self.Shop_Table_PATH, prim_path=prim_path)
        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"table_{i}", collision=True))

        prim_path = f"/World/Env_{i}/tooling_plate_{i}"
        add_reference_to_stage(usd_path=self.Tooling_Plate_PATH, prim_path=prim_path)
        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"tooling_plate_{i}", collision=True))

        prim_path = f"/World/Env_{i}/pipe_{i}"
        add_reference_to_stage(usd_path=self.Pipe_PATH, prim_path=prim_path)
        self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"pipe_{i}", collision=True))
        return


    def add_nuts_bolts_assets(self, i, env_offset):
        for bolt in range(1):
            # bolt[object_index]_[Env_index]
            prim_path = f"/World/Env_{i}/bolt{bolt}_{i}"
            add_reference_to_stage(usd_path=self.Bolt_PATH, prim_path=prim_path)
            self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"bolt{bolt}_{i}"))
        for nut in range(2):
            prim_path = f"/World/Env_{i}/nut{nut}_{i}"
            add_reference_to_stage(usd_path=self.Nut_PATH, prim_path=prim_path)
            self._world.scene.add(GeometryPrim(prim_path=prim_path, name=f"nut{nut}_{i}"))
        return


    def add_robot(self, i, env_offset):
        from omni.isaac.custom_franka import Franka3F, Franka2F

        self.ROBOT_PATH = "omniverse://localhost/Projects/ETRI/Manipulator/Franka/franka_2f85_screw.usd"

        robot = self._world.scene.add(
            Franka2F(
                prim_path=f"/World/Env_{i}/robot_{i}",
                usd_path=self.ROBOT_PATH,
                name=f"robot_{i}",
                # Table과의 충돌을 막기 위해 manual position 지정
                position=np.array([-0.7, 0.0, 0.0]) + env_offset, 
                orientation=np.array([1, 0, 0, 0]),
                attach_gripper=True,
            )
        )
        return


    def setup_scene(self):
        # super().setup_scene()
        self._setup_simulation()
        self._world.scene.add(XFormPrim(prim_path="/World/collisionGroups", name="collision_groups_xform"))
        
        self._world.scene.add_default_ground_plane()

        for i, env_offset in enumerate(self._env_offsets):

            self.add_xform(i, env_offset)
            self.add_robot(i, env_offset)
            # self.add_random_interactive_objects(i)
            self.add_table_assets(i, env_offset)
            self.add_nuts_bolts_assets(i, env_offset)


    def setup_robots(self, i, env_offset):
        # super().setup_robots(i)

        self._robot_position = np.array([0.269, 0.1778, 0.0])  # Gf.Vec3f(0.269, 0.1778, 0.0)
        robot_position = self._robot_position + env_offset
        robot = self._world.scene.get_object(f"robot_{i}")

        robot_pos = np.array(robot_position)
        robot_pos[2] = robot_pos[2] + self._table_height

        robot.set_world_pose(position=robot_pos)
        robot.set_default_state(position=robot_pos)
        robot.gripper.open()

        # robot = self._robots[i]
        # robot.gripper.open()
        # if self._gripper_type == "normal":
        #     kps = np.array([6000000.0, 600000.0, 6000000.0, 600000.0, 25000.0, 15000.0, 25000.0, 15000.0, 15000.0])
        #     kds = np.array([600000.0, 60000.0, 300000.0, 30000.0, 3000.0, 3000.0, 3000.0, 6000.0, 6000.0])
        #     robot.get_articulation_controller().set_gains(kps=kps, kds=kds, save_to_usd=True)
    
        controller = NutBoltController(
            name="nut_bolt_controller", 
            robot=robot,
            robot_type="franka",
        )

        self._controllers.append(controller)
        self._robots.append(robot)
        return
    

    def add_task(self, index, env_offset):
        super().add_task(index, env_offset)

        try:
            cur_robot = self._robots[index]
        except Exception as e:
            logging.error(f"[InsertionEnv] Failed to get robot at index {index}")
            return
        
        if self._task_type == "screw":
            # controller = NutBoltController(
            #     name=f"nut_bolt_controller_{index}", 
            #     robot=cur_robot,
            #     robot_type=self._robot_type
            # )

            self._nut_height = self._task_config["nut_height"]
            self._bolt_length = self._task_config["bolt_length"]

            self._gripper_to_nut_offset = self._task_config["gripper_to_nut_offset"]
            self._gripper_to_bolt_offset = self._task_config["gripper_to_bolt_offset"]

            self._top_of_bolt = (
                np.array([0.001, 0.0, self._bolt_length + (self._nut_height / 2)]) + self._gripper_to_bolt_offset
            )

        # self._controllers.append(controller)
    def add_table_property(self, i, env_offset):
        self._table_height = 0.0
        self._table_scale = 0.01
        self._table_position = np.array([0.5, 0.0, 0.0])  # Gf.Vec3f(0.5, 0.0, 0.0)
        self._tooling_plate_offset = np.array([0.0, 0.0, 0.0])
        self._pipe_pos_on_table = np.array([0.2032, 0.381, 0.0])

        self._table_local_position = self._table_position + env_offset
        self._table_ref_geom = self._world.scene.get_object(f"table_{i}")
        self._table_ref_geom.set_local_scale(np.array([self._table_scale]))
        self._table_ref_geom.set_world_pose(position=self._table_local_position)
        self._table_ref_geom.set_default_state(position=self._table_local_position)
        lb = self._world.scene.compute_object_AABB(name=f"table_{i}")
        zmin = lb[0][2]
        zmax = lb[1][2]
        self._table_local_position[2] = -zmin
        self._table_height = zmax
        self._table_ref_geom.set_collision_approximation("none")
        self._convexIncludeRel.AddTarget(self._table_ref_geom.prim_path)

        ##tooling_plate
        self._tooling_plate_geom = self._world.scene.get_object(f"tooling_plate_{i}")
        self._tooling_plate_geom.set_local_scale(np.array([self._table_scale]))
        lb = self._world.scene.compute_object_AABB(name=f"tooling_plate_{i}")
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
        self._pipe_geom = self._world.scene.get_object(f"pipe_{i}")
        self._pipe_geom.set_local_scale(np.array([self._table_scale]))
        lb = self._world.scene.compute_object_AABB(name=f"pipe_{i}")
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


    def add_nuts_and_bolt_property(self, i, env_offset):
        angle_delta = np.pi * 2.0 / 6
        for j in range(1):
            self._bolt_geom = self._world.scene.get_object(f"bolt{j}_{i}")

            bolt_pos = np.array([0.8132, 0.381, 0.91934])
            self._bolt_geom.set_world_pose(position=bolt_pos)
            self._bolt_geom.set_default_state(position=bolt_pos)

            print(f"self._bolt_geom.prim_path: {self._bolt_geom.prim_path}")
            self._boltMeshIncludeRel.AddTarget(self._bolt_geom.prim_path)


        nut_init_poses = np.array(
            [
                [0.6, 0.0, 0.85, 1.0, 0.0, 0.0, 0.0],
                [0.6, -0.1, 0.85, 1.0, 0.0, 0.0, 0.0],
            ], dtype=np.float32
        )

        for nut_idx in range(2):
            nut_pos = nut_init_poses[nut_idx, :3].copy() + env_offset

            self._nut_geom = self._world.scene.get_object(f"nut{nut_idx}_{i}")
            self._nut_geom.set_world_pose(position=np.array(nut_pos.tolist()))
            self._nut_geom.set_default_state(position=np.array(nut_pos.tolist()))

            print(f"self._nut_geom.prim_path: {self._nut_geom.prim_path}")
            self._convexIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_Convex")
            self._nutMeshIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_SDF")
            
            physxAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(self._nut_geom.prim.GetPrim())
            physxAPI.CreateSleepThresholdAttr().Set(0.0)

    # def _setup_materials(self):
    #     self._bolt_physics_material = PhysicsMaterial(
    #         prim_path="/World/PhysicsMaterials/BoltMaterial",
    #         name="bolt_material_physics",
    #         static_friction=0.2,
    #         dynamic_friction=0.2,
    #     )
    #     self._nut_physics_material = PhysicsMaterial(
    #         prim_path="/World/PhysicsMaterials/NutMaterial",
    #         name="nut_material_physics",
    #         static_friction=0.2,
    #         dynamic_friction=0.2,
    #     )
    #     self._robot_finger_physics_material = PhysicsMaterial(
    #         prim_path="/World/PhysicsMaterials/RobotFingerMaterial",
    #         name="robot_finger_material_physics",
    #         static_friction=0.7,
    #         dynamic_friction=0.7,
    #     )

    async def setup_post_load(self):
        # await super().setup_post_load()
        self._world.scene.enable_bounding_boxes_computations()
        # await self._setup_materials()

        # Here 
        for i, env_offset in enumerate(self._env_offsets):
            await self.add_random_lights(env_offset)

            self.add_table_property(i, env_offset)
            self.add_nuts_and_bolt_property(i, env_offset)
            self.setup_robots(i, env_offset)
            self.add_task(i, env_offset)

        return
    

    def forward(self):

        for i, robot in enumerate(self._robots):

            if self._save_count == 0:
                robot.gripper.open()

            if self._controllers[i].is_paused():
                self._controllers[i].reset(robot)

            # if self._controllers[i]._i < min(self._num_nuts, self._num_bolts):
                # nut_geom = self._world.scene.get_object(f"Env_{i}_nut0")
                # bolt_geom = self._world.scene.get_object(f"Env_{i}_bolt0")
                
            if self._controllers[i]._i < min(2, 6):
                # Here 
                nut_geom = self._world.scene.get_object(f"nut0_{i}")
                bolt_geom = self._world.scene.get_object(f"bolt0_{i}")

                nut_position, _ = nut_geom.get_world_pose()
                bolt_position, _ = bolt_geom.get_world_pose()

                initial_position = nut_position
                placing_position = bolt_position + self._top_of_bolt
                print(f"placing_position: {placing_position}")

                self._controllers[i].forward(
                    initial_picking_position=initial_position,
                    bolt_top=placing_position,
                    gripper_to_nut_offset=self._gripper_to_nut_offset,
                    x_offset=0.0,
                )

        self._save_count += 1
