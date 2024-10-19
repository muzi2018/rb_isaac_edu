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


import os

from omni.isaac.examples.base_sample import BaseSampleExtension
# from .replicator_basic_1 import ReplicatorBasic
from .replicator_basic_1_1 import ReplicatorBasic
# from .replicator_basic_2 import ReplicatorBasic
# from .replicator_basic_3 import ReplicatorBasic


class Extension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="RoadBalanceEdu",
            submenu_name="",
            name="ReplicatorBasic",
            title="ReplicatorBasic",
            doc_link="https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html",
            overview="This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            file_path=os.path.abspath(__file__),
            sample=ReplicatorBasic(),
        )
        return
