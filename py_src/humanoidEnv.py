"""******************************************************************************************
* HumARConoid-Mujoco
*
* Sim2Sim Transfer to Optimize Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid-MuJoCo.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

import torch
import numpy as np
import math

import mujoco
from mujoco import viewer

from model import MJCKIMANOID_MODEL_DIR
from py_src.policy import MJCKIMANOID_POLICY_DIR
from build import controller
from utils import detect_undesired_contact, name2id, detect_all_contacts

CONTROL = 1
RL = 2

BODY = 1
JOINT = 3
GEOM = 5

class kimanoid_env:
    def __init__(self) -> None:
        self.dof = 17
        self.model_path = self.model_path = f"{MJCKIMANOID_MODEL_DIR}/KIST_HUMANOID_TORSO/urdf/kist_humanoid3.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.num_commands = 3
        self.velocity_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.prev_action = np.zeros(self.dof, dtype=np.float32)
        self.proj_gravity = [0, 0, -1.0]
        self.torque = np.zeros(self.dof, dtype=np.float32)
        
        self.q_init = [0, 0, 0.852, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.q_init = [0, 0, 0.83, 1, 0, 0, 0, -0.0 * torch.pi / 180, -1.607 * torch.pi / 180, 8.367 * torch.pi / 180, 38.170 * torch.pi / 180, 30.0 * torch.pi / 180,
        #                0.0, 0.0, -0.0, -1.607 * torch.pi / 180, -8.367 * torch.pi / 180, -38.170 * torch.pi / 180, -30.0 * torch.pi / 180, -0.0, 0.0, 0, 0, 0.0]
        # self.q_init = * torch.pi / 180 * self.q_init
        self.qdot_init = np.zeros(self.dof + 6, dtype=np.float32)
        
        self.rendering = True
        self.viewer = None

    def reset(self):
        self.velocity_commands = np.zeros(self.dof, dtype=np.float32)
        self.prev_action = np.zeros(self.dof, dtype=np.float32)
        
        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init
        
        mujoco.mj_step(self.model, self.data)
        
        
        obs = self._observation()
        
        if self.rendering:
            self.render()
            
        return obs

    def step(self, action):
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32).flatten()
        prev_action = action.copy() # isaaclab order
        
        # action = [-0.0 * torch.pi / 180, -1.607 * torch.pi / 180, 8.367 * torch.pi / 180, 38.170 * torch.pi / 180, 30.0 * torch.pi / 180,
        #                0.0, 0.0, -0.0, -1.607 * torch.pi / 180, -8.367 * torch.pi / 180, -38.170 * torch.pi / 180, -30.0 * torch.pi / 180, -0.0, 0.0, 0, 0, 0.0]
        action = self.jnt_isaaclab2mujoco(action) # mujoco order
        self.pd_control(action)
        # self.torque = [1.54841, -1.53545, 1.80953, -5.08482, -5.4213, 1.57883, 0.781247, 1.61632, -1.49682, -2.60416, 7.62167, 8.59242, 2.11085, -1.18872, 0.084393, 0.00726544, 3.58722]
        
        for i in range(self.dof):
            self.data.ctrl[i] = self.torque[i]
        # self.data.ctrl[0] = 10
        for i in range(2):
            mujoco.mj_step(self.model, self.data)
        
        # print(self.data.ctrl.shape)
        # print(self.data.qfrc_actuator)
        
        obs = self._observation()

        self.prev_action = prev_action

        if self.rendering:
            self.render()

        return obs

    def _observation(self):
        self.proj_gravity = self.quat_rotate_inverse(self.data.qpos[3:7])
        self.velocity_commands = [0.5, 0, 0]
        
        # self.data.qvel[0:3] # base_lin_vel(3)
        # self.data.qvel[3:6] # base_ang_vel(3)
        # self.data.proj_gravity # projected_gravity(3)
        # self.velocity_commands # velocity_commands(3)
        # self.data.qpos[7:] # joint_pos(17)
        # self.data.qvel[6:] # joint_vel(17)
        # self.prev_action # actions(17)
        
        base_lin_vel = self.data.qvel[0:3]                          # (3,)
        base_ang_vel = self.data.qvel[3:6]                          # (3,)
        proj_gravity = self.proj_gravity                            # (3,)
        velocity_commands = self.velocity_commands                  # (3,)
        joint_pos = self.jnt_mujoco2isaaclab(self.data.qpos[7:])    # (17,)
        joint_vel = self.jnt_mujoco2isaaclab(self.data.qvel[6:])    # (17,)
        prev_action = self.jnt_mujoco2isaaclab(self.prev_action)    # (17,)

        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            proj_gravity,
            velocity_commands,
            joint_pos,
            joint_vel,
            prev_action
        ]).astype(np.float32)

        # print(f"Concatenated data shape: {obs.shape}")
        obs = obs.reshape(1, -1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        return obs_tensor.numpy()

    def _reward(self):
        pass

    def _done(self):
        pass

    def _info(self):
        print("num of joints: ", self.model.njnt)
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            # print(f"Joint ID: {joint_id}, Joint Name: {joint_name}")
        pass

    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
            self.viewer.cam.lookat = self.data.body('base_link').subtree_com
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 70
            # self.viewer.cam.lookat = [0.15, 0, 0.5]
            # self.viewer.cam.distance = 3
        else:
            self.viewer.sync()
            
    def quat_rotate_inverse(self, quat):
        """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

        Args:
            q: The quaternion in (w, x, y, z). Shape is (..., 4).
            v: The vector in (x, y, z). Shape is (..., 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (..., 3).
        """
        vec = [0, 0, -1]
        
        quat = torch.tensor(quat, dtype=torch.float32)
        vec = torch.tensor(vec, dtype=torch.float32)
        
        q_w = quat[0]
        q_vec = quat[1:]



        # quat = np.asarray(quat)
        # vec = np.asarray(vec)
        
        a = vec * (2.0 * q_w**2 - 1.0)
        b = torch.cross(q_vec, vec, dim=-1) * q_w * 2.0
        c = q_vec * torch.einsum("...i,...i->...", q_vec, vec) * 2.0
        return (a - b + c).numpy()
    
    def jnt_isaaclab2mujoco(self, q):
        ''' Mapping the joint order between IsaacLab and MuJoCo
            IsaacLab -> MuJoCo
            
        Args: joint order in Isaaclab
            'LLJ1', 'RLJ1', 'WJ1', 'LLJ2', 'RLJ2', 'WJ2', 'LLJ3', 'RLJ3', 'WJ3', 'LLJ4', 'RLJ4', 'LLJ5', 'RLJ5', 'LLJ6', 'RLJ6', 'LLJ7', 'RLJ7'
        
        Returns: joint order in MuJoCo
            'LLJ1', 'LLJ2', 'LLJ3', 'LLJ4', 'LLJ5', 'LLJ6', 'LLJ7', 'RLJ1', 'RLJ2', 'RLJ3', 'RLJ4', 'RLJ5', 'RLJ6', 'RLJ7', 'BWJ1', 'BWJ2', 'BWJ3'
        '''
        
        # Define mapping indices for IsaacLab -> MuJoCo
        isaaclab_to_mujoco_indices = [
            0, 3, 6, 9, 11, 13, 15,  # LLJ1 to LLJ7
            1, 4, 7, 10, 12, 14, 16,  # RLJ1 to RLJ7
            2, 5, 8  # WJ1 to WJ3 (Mapped to BWJ1 to BWJ3)
        ]
        
        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)
        
        # Reorder the joints using the mapping indices
        q_mujoco = q[isaaclab_to_mujoco_indices]
        
        return q_mujoco
    
    def jnt_mujoco2isaaclab(self, q):
        ''' Mapping the joint order between IsaacLab and MuJoCo
            MuJoCo -> IsaacLab
            
        Args: joint order in MuJoCo
            'LLJ1', 'LLJ2', 'LLJ3', 'LLJ4', 'LLJ5', 'LLJ6', 'LLJ7', 'RLJ1', 'RLJ2', 'RLJ3', 'RLJ4', 'RLJ5', 'RLJ6', 'RLJ7', 'BWJ1', 'BWJ2', 'BWJ3'
        
        Returns: joint order in Isaaclab
            'LLJ1', 'RLJ1', 'WJ1', 'LLJ2', 'RLJ2', 'WJ2', 'LLJ3', 'RLJ3', 'WJ3', 'LLJ4', 'RLJ4', 'LLJ5', 'RLJ5', 'LLJ6', 'RLJ6', 'LLJ7', 'RLJ7'
        '''
        
        # Define mapping indices for MuJoCo -> IsaacLab
        mujoco_to_isaaclab_indices = [
            0, 7, 14,  # 'LLJ1', 'RLJ1', 'BWJ1' -> 'LLJ1', 'RLJ1', 'WJ1'
            1, 8, 15,  # 'LLJ2', 'RLJ2', 'BWJ2' -> 'LLJ2', 'RLJ2', 'WJ2'
            2, 9, 16,  # 'LLJ3', 'RLJ3', 'BWJ3' -> 'LLJ3', 'RLJ3', 'WJ3'
            3, 10,     # 'LLJ4', 'RLJ4' -> 'LLJ4', 'RLJ4'
            4, 11,     # 'LLJ5', 'RLJ5' -> 'LLJ5', 'RLJ5'
            5, 12,     # 'LLJ6', 'RLJ6' -> 'LLJ6', 'RLJ6'
            6, 13      # 'LLJ7', 'RLJ7' -> 'LLJ7', 'RLJ7'
        ]
        
        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)
        
        # Reorder the joints using the mapping indices
        q_isaaclab = q[mujoco_to_isaaclab_indices]
        
        return q_isaaclab
    
    def pd_control(self, des_position):
        current_position = self.data.qpos[7:]
        current_velocity = self.data.qvel[6:]
        
        for i in range(self.dof):
            if i <= 3 or 7 <= i <= 10:
                kp = 150
                kd = 5
            else:
                kp = 20
                kd = 2
                
            self.torque[i] = kp * (des_position[i] - current_position[i]) + kd * (0 - current_velocity[i])
            
        return self.torque


class g1_env:
    def __init__(self) -> None:
        self.dof = 37
        self.model_path = self.model_path = f"{MJCKIMANOID_MODEL_DIR}/G1/g1.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjOption()
        self.model.opt.timestep = 0.002

        self.policy_path = f"{MJCKIMANOID_POLICY_DIR}/g1/policy.pt"
        self.policy = torch.jit.load(self.policy_path)

        self.decimation = 4

        self.control_mode = CONTROL
        self.num_commands = 3
        self.velocity_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.prev_action = np.zeros(self.dof, dtype=np.float32)
        self.proj_gravity = [0, 0, -1.0]
        self.torque = np.zeros(self.dof, dtype=np.float32)
        self.done = False
        self.time_done = False

        self.q_default = [
            -0.2, 0., 0., 0.42, -0.23, 0.,
            -0.2, 0., 0., 0.42, -0.23, 0.,
            0.,
            0.35, 0.16, 0., 0.87, 0., 0.,
            1., 0.52, 0., 0., 0., 0.,
            0.35, -0.16, 0., 0.87, 0., 0.,
            -1., -0.52, 0., 0., 0., 0.
        ]
        self.q_init = [
            0, 0, 0.753, 1, 0, 0, 0,
            -0.2, 0., 0., 0.42, -0.23, 0.,
            -0.2, 0., 0., 0.42, -0.23, 0.,
            0.,
            0.35, 0.16, 0., 0.87, 0., 0.,
            1., 0.52, 0., 0., 0., 0.,
            0.35, -0.16, 0., 0.87, 0., 0.,
            -1., -0.52, 0., 0., 0., 0.
        ]
        self.q_des = self.q_default.copy()
        self.qdot_init = np.zeros(self.dof + 6, dtype=np.float32)
        
        self.q_range = [self.model.jnt_range[i + 1].tolist() for i in range(self.dof)] # except floating joint
        
        # define kp and kd values for each joint
        self.kp = [
            200, 150, 150, 200, 20, 20,
            200, 150, 150, 200, 20, 20,
            200, 40,  40,  40,  40, 40,
            40,  40,  40,  40, 40, 40, 40,
            40,  40,  40,  40, 40,
            40,  40,  40,  40, 40, 40, 40
        ]
        # self.kp = 4 / 5 * np.array(self.kp)
        
        self.kd = [
            5,  5,  5,  5,  2,  2,
            5,  5,  5,  5,  2,  2,
            5, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10
        ]
        
        self.action = np.zeros(self.dof, dtype=np.float32)
        self.action_scale = 1.
        
        self.ground_contact_id = 0
        undesired_contact_list = ["pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link"] # 2, 5, 20, 9, 24, 35
        self.undesired_contact_id = name2id(self.model, GEOM, undesired_contact_list)
        self.step_cnt = 0
        self.current_action = np.zeros(self.dof, dtype=np.float32)

        self.rendering = True
        self.viewer = None
        
        # print("========== initialize g1_env class ==========")

    def reset(self):
        self.done = False
        self.time_done = False
        self.control_mode = CONTROL
        self.velocity_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.prev_action = np.zeros(self.dof, dtype=np.float32)
        self.step_cnt = 0
        self.q_des = self.q_default.copy()

        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init
        
        self.torque = self.pd_control(self.q_default)
        
        for i in range(self.dof):
            self.data.ctrl[i] = self.torque[i]
        
        mujoco.mj_step(self.model, self.data)

        self.obs = self._observation()

        if self.rendering:
            self.render()

        self.velocity_commands = [0.6, 0, 0]

        return self.obs

    def step(self, action=None):
        if not self.done:
            self.torque = self.pd_control(self.q_des)
            
            for i in range(self.dof):
                self.data.ctrl[i] = self.torque[i]

            mujoco.mj_step(self.model, self.data)
            
            self.step_cnt += 1
            
            if self.step_cnt % self.decimation == 0:
                self.prev_action = self.jnt_mujoco2isaaclab(self.action * self.action_scale)
                self.obs = self._observation()
                
                self.action = self.choose_action(self.obs)
                self.q_des = self.action * self.action_scale + self.q_default

        done = self._done()
        
        if self.rendering:
            self.render()

        return self.obs, done

    def _observation(self):
        base_lin_vel = self.data.qvel[0:3]  # (3,)
        base_ang_vel = self.data.qvel[3:6]  # (3,)
        proj_gravity = self.get_gravity_orientation(self.data.qpos[3:7])  # (3,)
        velocity_commands = self.velocity_commands  # (3,)
        joint_pos = self.jnt_mujoco2isaaclab(self.data.qpos[7:] - self.q_default)  # (17,)
        joint_vel = self.jnt_mujoco2isaaclab(self.data.qvel[6:])  # (17,)
        prev_action = self.prev_action  # (17,)

        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            proj_gravity,
            velocity_commands,
            joint_pos,
            joint_vel,
            prev_action
        ]).astype(np.float32)

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        return obs_tensor
    
    def choose_action(self, obs_tensor):
        action = self.policy(obs_tensor).detach().numpy().squeeze() # isaaclab order
        return self.jnt_isaaclab2mujoco(action) # mujoco order

    def _reward(self):
        pass

    def _done(self):
        # reset done status
        done = False
        terminated_done = False
        time_done = False
        
        # collision
        is_ground_contact_list = detect_undesired_contact(self.data.contact, self.ground_contact_id, self.undesired_contact_id)
        if not len(is_ground_contact_list) == 0:
            terminated_done = True

        # time
        if self.time_done:
            time_done = True

        if terminated_done or time_done:
            done = True
        
        self.done = done
        
        return done

    def _info(self):
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_range = self.model.jnt_range[joint_id]
        pass

    def render(self):
        target_body = "pelvis"

        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 150
            self.viewer.cam.distance = 2
             
        target_position = self.data.body(target_body).xpos
        self.viewer.cam.lookat[:] = target_position
            
        self.viewer.sync()
    
    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def jnt_isaaclab2mujoco(self, q):
        ''' Mapping the joint order between IsaacLab and MuJoCo
            IsaacLab -> MuJoCo

        Args: joint order in Isaaclab
        Returns: joint order in MuJoCo

        '''

        # Define mapping indices for IsaacLab -> MuJoCo
        isaaclab_to_mujoco_indices = [
            0, 3, 7, 11, 15, 19, 1, 4,
            8, 12, 16, 20, 2, 5, 9, 13,
            17, 21, 25, 31, 35, 24, 30, 23, 29,
            6, 10, 14, 18, 22, 28, 34, 36, 27,
            33, 26, 32
        ]

        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)

        # Reorder the joints using the mapping indices
        q_mujoco = q[isaaclab_to_mujoco_indices]

        return q_mujoco

    def jnt_mujoco2isaaclab(self, q):
        ''' Mapping the joint order between IsaacLab and MuJoCo
            MuJoCo -> IsaacLab

        Args: joint order in MuJoCo
        Returns: joint order in Isaaclab

        '''

        # Define mapping indices for MuJoCo -> IsaacLab
        mujoco_to_isaaclab_indices = [
            0, 6, 12, 1, 7, 13, 25, 2, 8, 14, 26,
            3, 9, 15, 27, 4, 10, 16, 28, 5, 11,
            17, 29, 23, 21, 18, 35, 33, 30, 24, 22,
            19, 36, 34, 31, 20, 32
        ]

        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)

        # Reorder the joints using the mapping indices
        q_isaaclab = q[mujoco_to_isaaclab_indices]

        return q_isaaclab

    def pd_control(self, des_position):
        current_position = self.data.qpos[7:]
        current_velocity = self.data.qvel[6:]

        for i in range(self.dof):
            # Use corresponding kp and kd values for each joint
            kp = self.kp[i]
            kd = self.kd[i]
            
            # Calculate torque using PD control law
            self.torque[i] = kp * (des_position[i] - current_position[i]) + kd * (0 - current_velocity[i])

        return self.torque
    
    def jnt_normalize(self, q_unnormalized):
        normalized_values = [
            (joint_value - lower_limit) / (upper_limit - lower_limit)
            for (joint_value, (lower_limit, upper_limit)) in zip(q_unnormalized, self.q_range)
        ]

        return normalized_values

    def jnt_unnormalize(self, q_normalized):
        unnormalized_values = [
            normalized_value * (upper_limit - lower_limit) + lower_limit
            for (normalized_value, (lower_limit, upper_limit)) in zip(q_normalized, self.q_range)
        ]

        return unnormalized_values


class g1_kist_env:
    def __init__(self) -> None:
        self.dof = 12 # for obs
        self.action_dof = 12 # for action
        self.control_dof = 12 # for step
        self.model_path = self.model_path = f"{MJCKIMANOID_MODEL_DIR}/G1/g1_kist.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjOption()
        self.model.opt.timestep = 0.002

        self.policy_path = f"{MJCKIMANOID_POLICY_DIR}/g1_kist/policy.pt"
        self.policy = torch.jit.load(self.policy_path)

        self.decimation = 4

        # self.control_mode = CONTROL
        self.num_commands = 3
        self.velocity_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dof, dtype=np.float32)
        self.proj_gravity = [0, 0, -1.0]
        self.torque = np.zeros(self.control_dof, dtype=np.float32)
        self.done = False
        self.time_done = False

        self.q_default = [
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            # 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
        ]
        self.q_init = [
            0, 0, 0.793, 1, 0, 0, 0,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            # 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
        ]
        # self.q_des = self.q_default.copy()
        self.q_des = [
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            # 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
        ]
        self.qdot_init = np.zeros(len(self.q_default) + 6, dtype=np.float32)

        self.q_range = [self.model.jnt_range[i + 1].tolist() for i in range(self.control_dof)]  # except floating joint

        # define kp and kd values for each joint
        self.kp = [
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            # 200, 200, 200,
            # 40, 40, 40, 40, 40, 40, 40,
            # 40, 40, 40, 40, 40, 40, 40
        ]

        self.kd = [
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            # 5, 5, 5,
            # 2, 2, 2, 2, 2, 2, 2,
            # 2, 2, 2, 2, 2, 2, 2
        ]

        self.action = np.zeros(self.action_dof, dtype=np.float32)
        self.action_scale = 1.

        self.ground_contact_id = name2id(self.model, GEOM, ["floor"])
        undesired_contact_list = ["pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link",
                                  "right_hip_yaw_link", "torso_link"]  # 2, 5, 20, 9, 24, 35
        self.undesired_contact_id = name2id(self.model, GEOM, undesired_contact_list)
        self.step_cnt = 0
        self.current_action = np.zeros(self.action_dof, dtype=np.float32)

        self.rendering = True
        self.viewer = None

        # print("========== initialize g1_env class ==========")

    def reset(self):
        self.done = False
        self.time_done = False
        # self.control_mode = CONTROL
        self.velocity_commands = np.zeros(self.num_commands, dtype=np.float32)
        self.prev_action = np.zeros(self.action_dof, dtype=np.float32)
        self.step_cnt = 0
        # self.q_des = self.q_default.copy()
        self.q_des = [
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            # 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 0., 0., 0.,
        ]
        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init

        self.torque = self.pd_control(self.q_des)

        for i in range(self.control_dof):
            self.data.ctrl[i] = self.torque[i]

        mujoco.mj_step(self.model, self.data)

        self.obs = self.observation()

        if self.rendering:
            self.render()

        self.velocity_commands = [0., 0, 0]

        return self.obs

    def step(self, action=None):
        if not self.done:
            self.torque = self.pd_control(self.q_des)

            for i in range(self.control_dof):
                self.data.ctrl[i] = self.torque[i]

            mujoco.mj_step(self.model, self.data)

            self.step_cnt += 1

            if self.step_cnt % self.decimation == 0:
                self.prev_action = self.jnt_mujoco2isaaclab(self.action * self.action_scale)
                self.obs = self.observation()

                self.action = self.choose_action(self.obs)
                q_des = self.action * self.action_scale + self.q_default[:12]
                self.q_des[:12] = self.limit_q_values(q_des)

                # print("변환된 값들:", self.q_des[:12])

        done = self._done()

        if self.rendering:
            self.render()

        return self.obs, done

    def observation(self):
        base_lin_vel = self.data.qvel[0:3]  # (3,)
        base_ang_vel = self.data.qvel[3:6]  # (3,)
        proj_gravity = self.get_gravity_orientation(self.data.qpos[3:7])  # (3,)
        velocity_commands = self.velocity_commands  # (3,)
        joint_pos = self.jnt_mujoco2isaaclab(self.data.qpos[7:] - self.q_default)  # (17,)
        joint_vel = self.jnt_mujoco2isaaclab(self.data.qvel[6:])  # (17,)
        prev_action = self.prev_action  # (12,)

        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            proj_gravity,
            velocity_commands,
            joint_pos,
            joint_vel,
            prev_action
        ]).astype(np.float32)

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        return obs_tensor

    def choose_action(self, obs_tensor):
        action = self.policy(obs_tensor).detach().numpy().squeeze()  # isaaclab order
        return self.jnt_isaaclab2mujoco(action)  # mujoco order

    def reward(self):
        pass

    def _done(self):
        # reset done status
        done = False
        terminated_done = False
        time_done = False

        # collision
        is_ground_contact_list = detect_undesired_contact(self.data.contact, self.ground_contact_id,
                                                          self.undesired_contact_id)
        if not len(is_ground_contact_list) == 0:
            terminated_done = True
        print(f"is_ground_contact_list:{is_ground_contact_list}")

        # time
        if self.time_done:
            time_done = True

        if terminated_done or time_done:
            done = True

        self.done = done

        return done

    def _info(self):
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_range = self.model.jnt_range[joint_id]
        pass

    def render(self):
        target_body = "pelvis"

        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 150
            self.viewer.cam.distance = 2

        target_position = self.data.body(target_body).xpos
        self.viewer.cam.lookat[:] = target_position

        self.viewer.sync()

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def jnt_isaaclab2mujoco(self, q):
        ''' Mapping the joint order between IsaacLab and MuJoCo
            IsaacLab -> MuJoCo

        Args: joint order in Isaaclab
        Returns: joint order in MuJoCo

        '''

        # Define mapping indices for IsaacLab -> MuJoCo
        if len(q) == 12:
            isaaclab_to_mujoco_indices = [
                0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11
                # 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11
            ]

        elif len(q) == 17:
            isaaclab_to_mujoco_indices = [
                0, 3, 6, 9, 11, 15, 1, 4, 7, 10, 12, 16, 2, 5, 8, 13, 14
                # 0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 15, 16, 5, 11
            ]

        elif len(q) == 29:
            isaaclab_to_mujoco_indices = [
                0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
                # 0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 15, 16, 5, 11
            ]

        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)

        # Reorder the joints using the mapping indices
        q_mujoco = q[isaaclab_to_mujoco_indices]

        return q_mujoco

    def jnt_mujoco2isaaclab(self, q):
        ''' Mapping the joint order between IsaacLab and MuJoCo
            MuJoCo -> IsaacLab

        Args: joint order in MuJoCo
        Returns: joint order in Isaaclab

        '''

        # Define mapping indices for MuJoCo -> IsaacLab
        if len(q) == 12:
            mujoco_to_isaaclab_indices = [
                # 0, 2, 4, 6, 8, 10, 1, 3, 5, 9, 7, 11
                0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11
            ]
        elif len(q) == 17:
            mujoco_to_isaaclab_indices = [
                # 0, 3, 6, 9, 11, 15, 1, 4, 7, 10, 12, 16, 2, 5, 8, 13, 14
                0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 15, 16, 5, 11
            ]
        elif len(q) == 29:
            mujoco_to_isaaclab_indices = [
                # 0, 3, 6, 9, 11, 15, 1, 4, 7, 10, 12, 16, 2, 5, 8, 13, 14
                0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
            ]

        # Convert input to NumPy array if not already
        q = np.array(q, dtype=np.float32)

        # Reorder the joints using the mapping indices
        q_isaaclab = q[mujoco_to_isaaclab_indices]

        return q_isaaclab

    def pd_control(self, des_position):
        current_position = self.data.qpos[7:]
        current_velocity = self.data.qvel[6:]

        for i in range(self.control_dof):
            # Use corresponding kp and kd values for each joint
            kp = self.kp[i]
            kd = self.kd[i]

            # Calculate torque using PD control law
            self.torque[i] = kp * (des_position[i] - current_position[i]) + kd * (0 - current_velocity[i])

        return self.torque

    def jnt_normalize(self, q_unnormalized):
        normalized_values = [
            (joint_value - lower_limit) / (upper_limit - lower_limit)
            for (joint_value, (lower_limit, upper_limit)) in zip(q_unnormalized, self.q_range)
        ]
        return normalized_values

    def jnt_unnormalize(self, q_normalized):
        unnormalized_values = [
            normalized_value * (upper_limit - lower_limit) + lower_limit
            for (normalized_value, (lower_limit, upper_limit)) in zip(q_normalized, self.q_range)
        ]
        return unnormalized_values

    def jnt_range_pi(self, angle):
        while angle > math.pi:
            angle -= math.pi
        while angle < - math.pi:
            angle += math.pi
        return angle

    def limit_q_values(self, q):
        limited_q = []

        for i in range(len(q)):
            min_val, max_val = self.q_range[:self.dof][i]
            if q[i] < min_val:
                limited_q.append(min_val)
            elif q[i] > max_val:
                limited_q.append(max_val)
            else:
                limited_q.append(q[i])
        return limited_q

