from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.etri_env import MultiPickandPlaceEnv
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)

class DefectEnv(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # Franka Env
        self._env_json = current_dir + "/example_franka_normal_bg.json"
        return


    def setup_scene(self):
        self._world = self.get_world()

        self.pp_env = MultiPickandPlaceEnv(
            world=self._world,
            json_file_path=self._env_json,
            verbose=True,
        )

        self.pp_env.setup_scene()
        return


    async def setup_post_load(self):
        await self.pp_env.setup_post_load()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return


    def physics_step(self, step_size):
        self.pp_env.forward()
        return


    async def setup_pre_reset(self):
        self.pp_env.setup_pre_reset()
        self._event = 0
        return
    

    def world_cleanup(self):
        self.pp_env.world_cleanup()
        self._world.pause()
        return