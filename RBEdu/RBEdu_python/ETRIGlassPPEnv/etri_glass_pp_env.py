from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.etri_env import MultiPickandPlaceEnv
import os

# Get the current working directory
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)


class GlassPPEnv(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # ur5 2f glass cup
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/ur5_2f_glass_cup.json"
        # ur5 2f heineken_glass
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/ur5_2f_heineken_glass.json"
        # ur5 3f glass cup
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/ur5_3f_glass_cup.json"

        # ur10 2f glass cup 
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/ur10_2f_glass_cup.json"
        # ur10 3f glass cup
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/ur10_3f_glass_cup.json"

        # zeus 2f glass cup
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/zeus_2f_glass_cup.json"
        # zeus 3f glass cup
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/zeus_3f_glass_cup.json"

        # franka normal glass cup 
        self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/franka_normal_glass_cup.json"
        # franka 2f glass cup
        # self._env_json = current_dir + "/franka_2f_glass_cup.json"
        # franka 3f glass cup
        # self._env_json = "/home/kimsooyoung/Documents/IssacSimTutorials/rb_issac_tutorial/RoadBalanceEdu/ETRIGlassPPEnv/franka_3f_glass_cup.json"
        return


    def setup_scene(self):
        self._world = self.get_world()
        
        self.lab_example = MultiPickandPlaceEnv(
            world=self._world,
            json_file_path=self._env_json,
            verbose=True,
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