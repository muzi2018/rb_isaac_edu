from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, UsdLux, PhysxSchema
import numpy as np
import logging

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.custom_controller import NutBoltController 
from ..multi_env_randomizer import MultiEnvRandomizer


class MultiScrewEnvJSON(MultiEnvRandomizer):
    def __init__(
            self,
            world=None,
            env_offsets: list = [[0.0, 0.0, 0.0]],
            verbose: bool = False,
            json_file_path: str = None,
        ) -> None:

        super().__init__(
            world, 
            json_file_path,
            verbose
        )

        # pipe and bolt parameters
        self._num_nuts = 2
        self._num_bolts = 1
        return


    def _setup_simulation(self):
        self._scene = PhysicsContext()

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


    def setup_scene(self):
        self._world.scene.enable_bounding_boxes_computations()
        self._setup_simulation()
        super().setup_scene()

        for i, env_offset in enumerate(self._env_offsets):

            for object_config in self._interactive_objects_config:
                object_name = object_config["name"]
        
                if "bolt" in object_name.lower():
                    self._bolt_geom = self._world.scene.get_object(f"Env_{i}_{object_name}")
                    logging.info(f"self._bolt_geom.prim_path: {self._bolt_geom.prim_path}")
                    self._boltMeshIncludeRel.AddTarget(self._bolt_geom.prim_path)
                
                if "nut" in object_name.lower():
                    self._nut_geom = self._world.scene.get_object(f"Env_{i}_{object_name}")
                    logging.info(f"self._nut_geom.prim_path: {self._nut_geom.prim_path}")
                    self._convexIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_Convex")
                    self._nutMeshIncludeRel.AddTarget(self._nut_geom.prim_path + "/M20_Nut_Tight_SDF")

                    physxAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(self._nut_geom.prim.GetPrim())
                    physxAPI.CreateSleepThresholdAttr().Set(0.0)

        return


    def setup_robots(self, index):
        cur_robot = self._robots[index]


    def add_task(self, index, env_offset):
        super().add_task(index, env_offset)

        try:
            cur_robot = self._robots[index]
        except Exception as e:
            logging.error(f"[InsertionEnv] Failed to get robot at index {index}")
            return
        
        controller = NutBoltController(
            name=f"nut_bolt_controller_{index}", 
            robot=cur_robot,
            robot_type=self._robot_type
        )

        self._nut_height = self._task_config["nut_height"]
        self._bolt_length = self._task_config["bolt_length"]

        self._gripper_to_nut_offset = self._task_config["gripper_to_nut_offset"]
        self._gripper_to_bolt_offset = self._task_config["gripper_to_bolt_offset"]
        self._initial_end_effector_orientation = np.array(self._task_config["initial_end_effector_orientation"])
        
        if "placing_x_offset" in self._task_config:
            self._placing_x_offset = self._task_config["placing_x_offset"]
        else:
            self._placing_x_offset = 0.0

        self._top_of_bolt = (
            np.array([0.001, 0.0, self._bolt_length + (self._nut_height / 2)]) + self._gripper_to_bolt_offset
        )
        self._controllers.append(controller)


    async def setup_post_load(self):
        await super().setup_post_load()
        return


    def forward(self):

        for i, robot in enumerate(self._robots):

            if self._save_count == 0:
                robot.gripper.open()

            if self._controllers[i].is_paused():
                self._controllers[i].reset(robot)

            if self._controllers[i]._i < min(self._num_nuts, self._num_bolts):

                nut_geom = self._world.scene.get_object(f"Env_{i}_nut0")
                bolt_geom = self._world.scene.get_object(f"Env_{i}_bolt0")

                nut_position, _ = nut_geom.get_world_pose()
                bolt_position, _ = bolt_geom.get_world_pose()

                initial_position = nut_position
                placing_position = bolt_position + self._top_of_bolt

                self._controllers[i].forward(
                    initial_picking_position=initial_position,
                    bolt_top=placing_position,
                    gripper_to_nut_offset=self._gripper_to_nut_offset,
                    initial_end_effector_orientation=self._initial_end_effector_orientation,
                    x_offset=self._placing_x_offset,
                )

        super().forward()

