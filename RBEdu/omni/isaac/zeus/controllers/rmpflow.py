from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.articulations import Articulation
import omni.isaac.motion_generation as mg
import carb

from pathlib import Path
import os


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:
        
        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        cur_file_path = str(Path(__file__).absolute())
        cur_dir_path = os.path.dirname(cur_file_path)
        self._desc_path = cur_dir_path + "/rmpflow/"
        self._urdf_path = cur_dir_path + "/urdf/fsrobo_r.urdf"

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=self._desc_path+"robot_descriptor.yaml",
            rmpflow_config_path=self._desc_path+"zeus_rmpflow_common.yaml",
            urdf_path=self._urdf_path,
            end_effector_frame_name="Link6",
            maximum_substep_size=0.00334
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )