# import h5py
import os


class RoboDataCollector():
    def __init__(
            self, 
            file_name, 
            # robot_prim
        ):

        self._file_name = file_name
        # self._robot_prim = robot_prim

        # print(f"[RoboDataCollector] File Name: {self._file_name}")
        # file_path = str(expanduser("~") + "/Documents/ETRI_DATA/" + self._file_name + ".hdf5")
        file_path = file_name
        file_name = file_name.split("/")[-1]
        folder_path = file_path.replace(file_name, "")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at: {folder_path}")
        else:
            print(f"Folder already exists at: {folder_path}")

        # self._f = h5py.File(file_path, 'w')
        self._f = None

        # self._group_f = self._f.create_group(f"{file_name}")


    async def setup_pre_reset(self):
        if self._f is not None:
            self._f.close()
            self._f = None
        elif self._f is None:
            print("Create new file for new data collection...")
            self.setup_dataset()

    def create_single_pp_data(
            self,
            name,
            robot,
            controller,
        ):

        joint_positions = robot.get_joint_positions()
        joint_velocities = robot.get_joint_velocities()
        applied_joint_efforts = robot.get_applied_joint_efforts()
        measured_joint_efforts = robot.get_measured_joint_efforts()
        measured_joint_forces = robot.get_measured_joint_forces()
        gripper_pos, gripper_ori = robot.gripper.get_world_pose()

        pp_event = controller.get_current_event()

        try:
            if self._f is not None:
                cur_group = self._f.create_group(name)
                cur_group.attrs["joint_positions"] = joint_positions
                cur_group.attrs["joint_velocities"] = joint_velocities
                cur_group.attrs["applied_joint_efforts"] = applied_joint_efforts
                cur_group.attrs["measured_joint_efforts"] = measured_joint_efforts
                cur_group.attrs["measured_joint_forces"] = measured_joint_forces
                cur_group.attrs["gripper_pos"] = gripper_pos
                cur_group.attrs["gripper_ori"] = gripper_ori
                cur_group.attrs["pp_event"] = pp_event
            elif self._f is None:
                print("Invalid Operation Data not saved")
        except Exception as e:
            print(e)
        finally:
            return

    def create_multi_pp_data(
            self,
            name,
            robot,
            controller,
        ):

        joint_positions = robot.get_joint_positions()
        joint_velocities = robot.get_joint_velocities()
        applied_joint_efforts = robot.get_applied_joint_efforts()
        measured_joint_efforts = robot.get_measured_joint_efforts()
        measured_joint_forces = robot.get_measured_joint_forces()
        gripper_pos, gripper_ori = robot.gripper.get_world_pose()

        pp_event = controller.get_pp_event()
        multi_pp_event = controller.get_current_event()

        try:
            if self._f is not None:
                cur_group = self._f.create_group(name)
                cur_group.attrs["joint_positions"] = joint_positions
                cur_group.attrs["joint_velocities"] = joint_velocities
                cur_group.attrs["applied_joint_efforts"] = applied_joint_efforts
                cur_group.attrs["measured_joint_efforts"] = measured_joint_efforts
                cur_group.attrs["measured_joint_forces"] = measured_joint_forces
                cur_group.attrs["gripper_pos"] = gripper_pos
                cur_group.attrs["gripper_ori"] = gripper_ori
                cur_group.attrs["pp_event"] = pp_event
                cur_group.attrs["multi_pp_event"] = multi_pp_event
            elif self._f is None:
                print("Invalid Operation Data not saved")
        except Exception as e:
            print(e)
        finally:
            return

    def create_dataset(
            self,
            name,
            data_dict,
        ):
        try:
            if self._f is not None:
                cur_group = self._f.create_group(f"{name}")
                for dset_name in data_dict:
                    cur_group.attrs[f"{dset_name}"] = data_dict[dset_name]
            elif self._f is None:
                print("Invalid Operation Data not saved")
        except Exception as e:
            print(e)
        finally:
            return
    
    def save_data(self, i=None):
        try:
            if i == None:
                i = 0
            if self._f is not None:
                self._f.close()
                print(f"[RoboDataCollector] Env {i} Data saved")
            elif self._f is None:
                print("[RoboDataCollector] Invalid Operation Data not saved")
        except Exception as e:
            print(e)
        finally:
            self._f = None

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
