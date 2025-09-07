from py_src.configs import ARC_G1_CONFIG_DIR
from typing import Union
import numpy as np
import time
import torch

import os
import csv
from datetime import datetime

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
        self.target_dof_pos = config.default_angles.copy()        
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        
        self.current_position = np.zeros(29, dtype=np.float32)

        self.lin_vel = np.array([0.0, 0.0, 0.0])
        self.prev_time = time.time()
        
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")  
        self.log_file = f"g1_gain_tuning_log_{current_time}.csv"
        
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time"] + [f"Default_{i}" for i in range(12, 29)] +
                            [f"Current_{i}" for i in range(12, 29)] +
                            [f"Error_{i}" for i in range(12, 29)])

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

    def move_to_desired_pos(self):
        print("Moving to desired pos.") # move robot -> joint initial pose to goal pose (interpolation)
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx # 29 dof (whole body)
        kps = self.config.kps + self.config.arm_waist_kps # 29 dof (whole body)
        kds = self.config.kds + self.config.arm_waist_kds # 29 dof (whole body)
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
            
            for i in range(len(self.config.leg_joint2motor_idx)):
                self.current_position[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                self.current_position[i+12] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].q
            
            error_pose = default_pos[:12] - self.current_position[:12].copy()
            
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time()] + default_pos[:12].tolist() +
                                self.current_position[:12].tolist() +
                                error_pose.tolist())
                
            print(f"default_pos: \n{default_pos[12:]}")
            print(f"current_pos: \n{self.current_position[12:].copy()}")
            print(f"error_pose: \n{default_pos[12:]-self.current_position[12:].copy()}")

    def desired_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0) # 29 dof (whole body)
        
        while self.remote_controller.button[KeyMap.A] != 1:
            try:
                for i in range(len(self.config.leg_joint2motor_idx)):
                    motor_idx = self.config.leg_joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                for i in range(len(self.config.arm_waist_joint2motor_idx)):
                    motor_idx = self.config.arm_waist_joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                self.send_cmd(self.low_cmd)
                time.sleep(self.config.control_dt)
                
                for i in range(len(self.config.leg_joint2motor_idx)):
                    self.current_position[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
                for i in range(len(self.config.arm_waist_joint2motor_idx)):
                    self.current_position[i+12] = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[i]].q

                error_pose = default_pos[:12] - self.current_position[:12].copy()

                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time()] + self.config.default_angles[12:].tolist() +
                                    self.current_position[12:].tolist() +
                                    error_pose.tolist())
                                    
                print(f"default_pos: \n{default_pos[12:]}")
                print(f"current_pos: \n{self.current_position[12:].copy()}")
                print(f"error_pose: \n{default_pos[12:]-self.current_position[12:].copy()}")
                
            except KeyboardInterrupt:
                    break


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("net", type=str, help="network interface", default="eth0")
    # parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    # args = parser.parse_args()

    # Load config
    config_path = f"{ARC_G1_CONFIG_DIR}/g1_gain_tuning.yaml"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, "eth0")

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()
	
    # Move to the default position
    controller.move_to_desired_pos()

    # Enter the default position state, press the A key to continue executing
    controller.desired_pos_state()

    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    
    print("Exit")
