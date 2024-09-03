from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.etri_env import MultiEnvRandomizer
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


class EnvExample(BaseSample):

    def __init__(self) -> None:
        super().__init__()
        self._env_json = current_dir + "/random_env_example.json"
        return


    def setup_scene(self):
        self._world = self.get_world()
        
        self.lab_example = MultiEnvRandomizer(
            world=self._world,
            json_file_path=self._env_json,
            verbose=True
        )

        self.lab_example.setup_scene()
        return


    async def setup_post_load(self):
        await self.lab_example.setup_post_load()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        return


    def physics_step(self, step_size):
        self.lab_example.forward()
        return


    async def setup_pre_reset(self):
        self.lab_example.setup_pre_reset()
        self._event = 0
        return
    

    def world_cleanup(self):
        self.lab_example.world_cleanup()
        self._world.pause()
        return