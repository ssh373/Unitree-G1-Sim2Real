from py_src.configs import ARC_G1_CONFIG_DIR
from typing import Union
import numpy as np
import time
import torch
import onnxruntime as ort

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

import copy
import onnxruntime as ort
from functools import partial
from torch import nn
from pathlib import Path

from class_timer import Timer


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        self.start_time = time.perf_counter()

        # Initialize the policy network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = torch.jit.load(config.policy_path).to(self.device)
        self.onnx_policy = ort.InferenceSession("policy/g1/policy.onnx")
        self.input_name = self.onnx_policy.get_inputs()[0].name
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        
        
        # TODO: Export onnx
        """
        export_device = "cpu"
        torch_model_src = self.policy.to(export_device)
        class ONNXExportWrapper(nn.Module):
            def __init__(self, model):
                super(ONNXExportWrapper, self).__init__()
                self.model = model                
            def forward(self, observations):
                return self.model(observations)            
        
        torch_model = ONNXExportWrapper(torch_model_src)
        torch_input = torch.tensor(self.obs).to(export_device)[None,:]
        torch_model(torch_input)  
        savename = Path('custom.onnx')
        savename.parent.mkdir(parents=True, exist_ok=True)
        onnx_program = torch.onnx.export(
            torch_model,
            torch_input,
            savename,
            opset_version=9,
            input_names=["obs"],
            output_names=["action"])
        ort_sess = ort.InferenceSession(savename)
        outputs = ort_sess.run(None, {"obs": self.obs[None,:]})
        print("Exported policy to", savename)
        print("mean abs difference", (np.abs(outputs - torch_model(torch_input).detach().numpy())).mean())
        """

    
        import os
        os.exit(1)
        
        
        self.obs_tensor = torch.zeros((1,config.num_obs), dtype=torch.float32).to(self.device)

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
            
        # reset data logging buffer
        self.buffer_cur_pos = []
        self.buffer_des_pos = []
        self.buffer_time = []
        self.buffer_dt = []

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
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.init_leg_kps + self.config.init_arm_waist_kps
        kds = self.config.init_leg_kds + self.config.init_arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
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
        # run_time = time.perf_counter()
        self.counter += 1
        
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
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
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1
        
        # self.buffer_cur_pos.append(qj_obs)
        # target_dof_pos_log = self.target_dof_pos.flatten()
        # self.buffer_des_pos.append(target_dof_pos_log)

        num_actions = self.config.num_actions
        # self.obs[:3] = ang_vel
        # self.obs[3:6] = gravity_orientation
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        # self.obs[9 : 9 + num_actions] = qj_obs
        # self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        # self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        # self.obs[9 + num_actions * 3] = sin_phase
        # self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        
        # # obs_tensor = torch.from_numpy(self.obs).to(self.device).unsqueeze(0)
        # obs_tensor = torch.tensor(self.obs).to(self.device).unsqueeze(0)
        # # obs_tensor = torch.tensor(self.obs, dtype=torch.float32).pin_memory().to(self.device).unsqueeze(0)
        
        # start_inf = time.perf_counter()
        # self.action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()
        # end_inf = time.perf_counter()
        
        with torch.no_grad():
            start=time.perf_counter()
            self.action = self.onnx_policy.run(None, {self.input_name: np.expand_dims(self.obs,axis=0)})
            # self.obs_tensor[:] = torch.from_numpy(self.obs).unsqueeze(0).to('cuda', non_blocking=True)
            # self.action = self.policy(self.obs_tensor).detach().cpu().numpy().squeeze()
            print(time.perf_counter()-start)

        # transform action to target_dof_pos
        # target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        # self.target_dof_pos = target_dof_pos
        
        self.target_dof_pos[0] = 0.5 * sin_phase - 0.1

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_dof_pos[i]
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
        
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     motor_idx = self.config.leg_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.init_leg_kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.init_leg_kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0
        # for i in range(len(self.config.arm_waist_joint2motor_idx)):
        #     motor_idx = self.config.arm_waist_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.init_arm_waist_kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.init_arm_waist_kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)
        
        # time.sleep(self.config.control_dt)
        
        # self.next_time += self.config.control_dt
        # sleep_time = self.next_time - time.perf_counter()

        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        # else:
        #     # print(f"Loop overran by {-sleep_time:.6f} sec")
        #     self.next_time = time.perf_counter()
        #     # pass

        # end_time = time.perf_counter()
        # elapsed_time = end_time - self.start_time

        # self.buffer_time.append(elapsed_time)
        # self.buffer_dt.append(end_time - run_time)


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("net", type=str, help="network interface", default="eth0")
    # # parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    # args = parser.parse_args()

    # Load config
    config_path = f"{ARC_G1_CONFIG_DIR}/g1.yaml"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, "eth0")

    controller = Controller(config)
    print(f"[CUDA] is_available?: {torch.cuda.is_available()}")
    print(f"model:\n{controller.policy.eval()}")
    
    dummy_input = torch.zeros(1, controller.config.num_obs).to(controller.device)
    controller.policy(dummy_input)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()
    
    controller.start_time = time.perf_counter()
    controller.next_time = time.perf_counter()

    tmr_cmd = Timer(_name='Command',_HZ=50,_MAX_SEC=np.inf,_VERBOSE=True)
    
    tmr_cmd.start()
    while True:
        try:
            if tmr_cmd.do_run(): # plot (50HZ)
                controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    
    controller.cur_pos_array = np.array(controller.buffer_cur_pos)
    controller.des_pos_array = np.array(controller.buffer_des_pos)
    controller.time_array = np.array(controller.buffer_time)
    controller.dt_array = np.array(controller.buffer_dt)
    
    np.savez('log_data_simple_tutorial.npz',
             cur_pos=controller.cur_pos_array,
             des_pos=controller.des_pos_array,
             time=controller.time_array,
             dt=controller.dt_array
            )
    
    print("Exit")
    