from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils  


from omni.isaac.examples.base_sample import BaseSample
from pxr import UsdGeom, Gf, UsdPhysics

from omni.isaac.universal_robots.controllers import PickPlaceController
from omni.isaac.universal_robots import UR10

import numpy as np
import omni

#### Example4 - Custom Objects with Joint Efforts
class HelloManip(BaseSample):

    def __init__(self) -> None:
        super().__init__()

        self._isaac_assets_path = get_assets_root_path()
        self.CUBE_URL = self._isaac_assets_path + "/Isaac/Props/Blocks/nvidia_cube.usd"

        self._sim_count = 0
        return
    
    def add_robot(self):
        self._ur10 = self._world.scene.add(
            UR10(
                prim_path="/World/UR10",
                name="UR10",
                attach_gripper=True,
            )
        )
        return

    def add_cube(self):
        self._cube = add_reference_to_stage(usd_path=self.CUBE_URL, prim_path=f"/World/NVIDIA_Cube")
        self._stage = omni.usd.get_context().get_stage()

        # modify cube mesh
        cube_mesh = UsdGeom.Mesh.Get(self._stage, "/World/NVIDIA_Cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.5, -0.5, 0.1))

        # modify cube mass
        cube_mass = UsdPhysics.MassAPI.Apply(get_current_stage().GetPrimAtPath("/World/NVIDIA_Cube"))
        cube_mass.CreateMassAttr().Set(0.1)
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        self.add_robot()
        self.add_cube()
        return

    async def setup_post_load(self):
        self._world = self.get_world()

        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._ur10.gripper,
            robot_articulation=self._ur10,
        )

        self._ur10.gripper.open()
        self._articulation_controller = self._ur10.get_articulation_controller()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        joint_indices = self._ur10._articulation_view._metadata.joint_indices
        
        self.joint_len = len(joint_indices)
        self._swapped_joint_indices = {value: key for key, value in joint_indices.items()}
        print(f"joint_indices: {joint_indices} / joint_len: {self.joint_len}")
        return

    def physics_step(self, step_size):

        self._sim_count += 1

        joints_state = self._ur10.get_joints_state()
        actions = self._controller.forward(
            picking_position=np.array([0.5, -0.5, 0.03]),
            placing_position=np.array([0.5, 0.5, 0.05]),
            current_joint_positions=joints_state.positions,
            end_effector_offset=np.array([0, 0, 0.035]),
            end_effector_orientation=euler_angles_to_quat(
                np.array([np.pi, np.pi/2, -np.pi/2])
            ),
        )

        if self._sim_count % 100 == 0:
            joint_efforts = self._ur10.get_measured_joint_forces()
            for i in range(self.joint_len):
                print(f"joint_efforts [{self._swapped_joint_indices[i]}]: {joint_efforts[i]}")

        if self._controller.is_done():
            self._world.pause()
        else:
            self._articulation_controller.apply_action(actions)
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._controller.reset()
        return

    def world_cleanup(self):
        return
