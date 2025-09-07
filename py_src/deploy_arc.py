from py_src.configs import ARC_G1_CONFIG_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)

        # Initializing process variables
        self.qj = np.zeros(12, dtype=np.float32)
        self.dqj = np.zeros(12, dtype=np.float32)
        self.action_isaac = np.zeros(10, dtype=np.float32)
        self.action_mujoco = np.zeros(10, dtype=np.float32)
        self.cmd_action = np.zeros(12, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        self.qj_leg_range = np.array(
            [[-2.5307, 2.8798], [-0.5236, 2.9671], [-2.7576, 2.7576], [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618],
             [-2.5307, 2.8798], [-2.9671, 0.5236], [-2.7576, 2.7576], [-0.087267, 2.8798], [-0.87267, 0.5236], [-0.2618, 0.2618]]
        )
        self.prev_time = time.time()

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
            # if [START] button be pressed, finish zero_torque_state and break
            # next step: move_to_default_pos

    def move_to_default_pos(self):
        print("Moving to default pos.") # move robot -> joint initial pose to goal pose (interpolation)
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx # 29 dof (whole body)
        kps = self.config.init_leg_kps + self.config.init_arm_waist_kps # 29 dof (whole body)
        kds = self.config.init_leg_kds + self.config.init_arm_waist_kds # 29 dof (whole body)
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0) # 29 dof (whole body)
        dof_size = len(dof_idx) # 29 dof (whole body)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.init_leg_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.init_leg_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.init_arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.init_arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)


    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)): # get joint (motor) state for obs
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_joints = 12
        num_actions = self.config.num_actions # 10

        self.obs[:3] = ang_vel # 3
        self.obs[3:6] = gravity_orientation # 3
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd # 3
        self.obs[9:9 + num_joints] = self.jnt_mujoco2isaaclab(qj_obs) # 12
        self.obs[9 + num_joints:9 + num_joints * 2] = self.jnt_mujoco2isaaclab(dqj_obs) # 12
        self.obs[9 + num_joints * 2:9 + num_joints * 2 + num_actions] = self.action_isaac # 10

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action_isaac = self.policy(obs_tensor).detach().numpy().squeeze() # 10
        self.action_mujoco = self.jnt_isaaclab2mujoco(self.action_isaac) # 10

        # add roll joint cmds
        self.cmd_action = np.insert(self.action_mujoco, 5, 0)
        self.cmd_action = np.insert(self.cmd_action, 11, 0)

        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.cmd_action * self.config.action_scale
        target_dof_pos = self.limit_q_values(target_dof_pos)

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.init_arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.init_arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)

    def jnt_isaaclab2mujoco(self, q):
        ''' 
            Mapping the joint order between IsaacLab and MuJoCo
            IsaacLab -> MuJoCo
            Args: joint order in Isaaclab
            Returns: joint order in MuJoCo
        '''
        # Define mapping indices for IsaacLab -> MuJoCo
        if len(q) == 12:
            isaaclab_to_mujoco_indices = [
                0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11
            ]
        elif len(q) == 10:
            isaaclab_to_mujoco_indices = [
                0, 2, 4, 6, 8, 1, 3, 5, 7, 9
            ]

        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)
        # Reorder the joints using the mapping indices
        q_mujoco = q[isaaclab_to_mujoco_indices]

        return q_mujoco

    def jnt_mujoco2isaaclab(self, q):
        '''
            Mapping the joint order between IsaacLab and MuJoCo
            MuJoCo -> IsaacLab
            Args: joint order in MuJoCo
            Returns: joint order in Isaaclab
        '''
        # Define mapping indices for MuJoCo -> IsaacLab
        if len(q) == 12:
            mujoco_to_isaaclab_indices = [
                0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11
            ]
        elif len(q) == 10:
            mujoco_to_isaaclab_indices = [
                0, 6, 1, 7, 2, 8, 3, 9, 4, 5
            ]

        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)
        # Reorder the joints using the mapping indices
        q_isaaclab = q[mujoco_to_isaaclab_indices]

        return q_isaaclab

    def limit_q_values(self, q):
        limited_q = []

        for i in range(len(q)):
            min_val, max_val = self.qj_leg_range[:12][i]
            if q[i] < min_val:
                limited_q.append(min_val)
            elif q[i] > max_val:
                limited_q.append(max_val)
            else:
                limited_q.append(q[i])
        return limited_q


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("net", type=str, help="network interface", default="eth0")
    # parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    # args = parser.parse_args()

    # Load config
    config_path = f"{ARC_G1_CONFIG_DIR}/g1_arc.yaml"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, "eth0")

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
