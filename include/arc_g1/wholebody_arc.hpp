/******************************************************************************************
* HumARConoid-Sim2Real
*
* Sim2Real Transfer to Optimize Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid-Sim2Real.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi (Jennifer) */

#ifndef WHOLEBODY_ARC_HPP
#define WHOLEBODY_ARC_HPP

#include <cmath>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <string>
#include <functional>
#include <iostream>
#include <fstream>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "arc_g1/gamepad.hpp"
#include "arc_g1/utils.hpp"
#include "arc_g1/loop_freq_monitor.hpp"

// DDS includes
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL includes
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// Estimation (Odometer)
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

// onnxruntime
#include <onnxruntime_cxx_api.h>

// Topics
static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string GO_ODOM_TOPIC = "rt/odommodestate"; // high frequency

// namespace
using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class WholeBodyRL {
 public:
  WholeBodyRL(std::string networkInterface, const std::string& model_path);
  ~WholeBodyRL();

  /*****************************************************************************
  ** Define variables
  *****************************************************************************/
  static const int G1_NUM_MOTOR = 29;
  std::atomic<bool> should_exit_ = false;
  std::atomic<bool> logging_active_ = false;

  /*****************************************************************************
  ** Define functions
  *****************************************************************************/

 private:
  /*****************************************************************************
  ** Define publisher & subscriber & thread
  *****************************************************************************/
  ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;

  ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> estimate_state_subscriber;

  ThreadPtr command_writer_ptr_, control_thread_ptr_;
  std::thread monitor_thread_;
  std::thread logger_thread_;

  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;

  /*****************************************************************************
  ** Define enum class
  *****************************************************************************/
  enum class State 
  {
    WAIT_FOR_INIT_COMMAND,
    MOVING_TO_DEFAULT,
    HOLDING_DEFAULT,
    WAIT_FOR_POLICY_COMMAND,
    RL_POLICY_ACTIVE,
    RL_POLICY_WAVE_HAND,
    DAMPING_STATE
  };
  
  enum class Mode 
  {
    PR = 0,  // Series Control for Pitch/Roll Joints
    AB = 1   // Parallel Control for A/B Joints
  };

  enum G1JointIndex 
  {
    LeftHipPitch = 0,
    LeftHipRoll = 1,
    LeftHipYaw = 2,
    LeftKnee = 3,
    LeftAnklePitch = 4,
    LeftAnkleRoll = 5,
    RightHipPitch = 6,
    RightHipRoll = 7,
    RightHipYaw = 8,
    RightKnee = 9,
    RightAnklePitch = 10,
    RightAnkleRoll = 11,
    WaistYaw = 12,
    WaistRoll = 13,        // NOTE INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14,       // NOTE INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15,
    LeftShoulderRoll = 16,
    LeftShoulderYaw = 17,
    LeftElbow = 18,
    LeftWristRoll = 19,
    LeftWristPitch = 20,   // NOTE INVALID for g1 23dof
    LeftWristYaw = 21,     // NOTE INVALID for g1 23dof
    RightShoulderPitch = 22,
    RightShoulderRoll = 23,
    RightShoulderYaw = 24,
    RightElbow = 25,
    RightWristRoll = 26,
    RightWristPitch = 27,  // NOTE INVALID for g1 23dof
    RightWristYaw = 28     // NOTE INVALID for g1 23dof
  };

  /*****************************************************************************
  ** Define variables
  *****************************************************************************/
  float time_, time_abs;
  float control_dt_;  // [2ms]: 500hz
  float command_dt_;  // [20ms]: 50hz
  float duration_;    // [5 s]
  int counter_;
  Mode mode_pr_;
  uint8_t mode_machine_;
  std::atomic<bool> safe_freq_ = false;

  int high_odom_cnt_ = 0; // HERE

  static const std::array<float, G1_NUM_MOTOR> Kp;
  static const std::array<float, G1_NUM_MOTOR> Kd;

  Gamepad gamepad_;
  REMOTE_DATA_RX rx_;

  LoopFrequencyMonitor control_freq_monitor;
  LoopFrequencyMonitor command_writer_freq_monitor;

  // onnxruntime
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::SessionOptions session_options_;

  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> output_names;

  std::vector<float> input_data;
  std::array<int64_t, 2> input_shape;
  Ort::Value input_tensor_{nullptr};
  
  // create hidden state, cell state tensors
  std::vector<float> h_in_data;
  std::vector<float> c_in_data;
  std::array<int64_t, 3> hidden_shape; // LSTM: (num_layers, batch_size, hidden_size)

  // isaaclab2mujoco
  static constexpr std::array<int, 12> isaaclab2mujoco = 
  {
    0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11
  };

  // mujoco2isaaclab
  static constexpr std::array<int, 12> mujoco2isaaclab = 
  {
    0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11
  };

  static constexpr std::array<float, 29> joint_pos_min = 
  {
    -2.5307, -0.5236, -2.7576, -0.15, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.15, -0.87267, -0.2618,
    -0.75, -0.75, -0.75,
    -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
    -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443
  };

  static constexpr std::array<float, 29> joint_pos_max = 
  {
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    0.75, 0.75, 0.75,
    2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
    2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443
  };

  static constexpr std::array<float, 29> torque_limit = 
  {
    200, 200, 200, 200, 50, 50,
    200, 200, 200, 200, 50, 50,
    150, 150, 150,
    60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60
  };

  float joint_vel_limit = 7.f;

  // control state
  std::mutex cout_mutex;
  State state_ = State::WAIT_FOR_INIT_COMMAND;

  // RL policy
  int cnt = 0;
  // float period = 0.8;
  float stride_a = 8.0e-7;
  float stride_b = 1.0;
  float eps = 1e-07;
  size_t start_idx = 9;
  static constexpr size_t num_obs = 47;  // 47
  static constexpr size_t num_action = 12;
  std::vector<float> input_tensor_values_;
  std::array<int64_t, 2> input_shape_ = {1, 47};  // 47
  std::array<float, num_action> rl_action_ = {};

  std::array<float, 2> arm_swing_motion = {};

  // RL_POLICY_WAVE_HAND
  std::array<float, 4> arm_swing_wave_hand_goal = {0.1, -1.75, -1.5, -0.5};
  float command_speed = 0.f;
  float wave_duration = 12.f;
  float return_duration = 8.f;
  float amp = 0.4;
  float freq = 0.5;
  float omega = 2 * M_PI * freq;

  /*****************************************************************************
  ** Define functions
  *****************************************************************************/
  void InitPublisher();
  void InitSubscriber();
  void InitThread();
  void MonitorThread();
  void LoggerThread();
  void LoadYamlConfig(const std::string& config_yaml_path);
  void PrintYamlConfig();
  void CheckExit();
  void Global2NewWorld();
  std::array<float, 16> NewWorld2Base();
  void LoadOnnxModel();
  void imuTorsoHandler(const void *message);
  void LowStateHandler(const void *message);
  /*high frequency message handler function for subscriber*/
  void HighFreOdomMessageHandler(const void* messages);
  void LowCommandWriter();
  std::array<float, 3> GetGravityOrientation(const std::array<float, 4>& q);
  std::array<float, 3> Quat2RPY(const std::array<float, 4>& q);
  std::vector<float> GetMockObservation();
  std::vector<float> GetObservation();
  std::array<float, 12> RunInference();
  std::array<float, 2> arm_swing_action(float leg_phase, float speed, float amplitude);
  void Control();

  /*****************************************************************************
  ** Define structure & data buffer
  *****************************************************************************/
  struct ImuState
  {
    std::array<float, 3> rpy = {};
    std::array<float, 3> omega = {};
    std::array<float, 4> quat = {};
  };

  struct MotorCommand
  {
    std::array<float, G1_NUM_MOTOR> q_target = {};
    std::array<float, G1_NUM_MOTOR> dq_target = {};
    std::array<float, G1_NUM_MOTOR> kp = {};
    std::array<float, G1_NUM_MOTOR> kd = {};
    std::array<float, G1_NUM_MOTOR> tau_ff = {};
  };

  struct MotorState
  {
    std::array<float, G1_NUM_MOTOR> q = {};
    std::array<float, G1_NUM_MOTOR> dq = {};
    std::array<float, G1_NUM_MOTOR> tau = {};
  };

  struct OdometerState
  {
    std::array<float, 3> position = {};
    std::array<float, 3> velocity = {};
    std::array<float, 3> euler = {};

    float yaw_speed;

    std::array<float, 4> quaternion = {};
  };

  struct RLAction
  {
    std::array<float, num_action> rl_action = {};
  };

  struct YamlConfig
  {
    std::string model_path;
    std::array<float, G1_NUM_MOTOR> default_pos = {};
    std::array<float, G1_NUM_MOTOR> init_kp = {};
    std::array<float, G1_NUM_MOTOR> init_kd = {};
    std::array<float, G1_NUM_MOTOR> rl_kp = {};
    std::array<float, G1_NUM_MOTOR> rl_kd = {};
    float ang_vel_scale;
    float dof_pos_scale;
    float dof_vel_scale;
    float action_scale;
    std::array<float, 3> cmd_scale = {};
    std::array<float, 3> max_cmd = {};
    size_t num_actions;
    size_t num_obs;
  };

  // create data buffer
  arc_g1::DataBuffer<ImuState> imu_state_buffer_;
  arc_g1::DataBuffer<MotorCommand> motor_command_buffer_;
  arc_g1::DataBuffer<MotorState> motor_state_buffer_;
  arc_g1::DataBuffer<OdometerState> odom_state_buffer_;
  arc_g1::DataBuffer<RLAction> rl_action_buffer_;
  YamlConfig cfg;

  // unitree_go::msg::dds_::SportModeState_ estimator_state{};

  inline std::array<float, 4> normalize_quat(std::array<float, 4> q) noexcept
  {
    const float n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (n2 > 0.0f) {
        const float invn = 1.0f / std::sqrt(n2);
        return { q[0]*invn, q[1]*invn, q[2]*invn, q[3]*invn };
    }
    return { 1.0f, 0.0f, 0.0f, 0.0f };
  }

  inline std::array<float, 16> mat4_mul(const std::array<float, 16>& A,
                                        const std::array<float, 16>& B) noexcept
  {
    std::array<float, 16> C{};

    for (int i = 0; i < 4; ++i) {        // row
        for (int j = 0; j < 4; ++j) {    // col
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += A[i*4 + k] * B[k*4 + j];
            }
            C[i*4 + j] = sum;
        }
    }

    return C;
  }

  std::array<float, 16> transf_global2nworld = {};
  std::array<float, 16> transf_nworld2base = {};
  float yaw_rel;
  float p_y;

  // struct Mat4 
  // {
  //   // row-major 4x4
  //   std::array<float, 16> m{};
  //   static Mat4 identity() noexcept {
  //     Mat4 T{};
  //     T.m = {1,0,0,0,
  //             0,1,0,0,
  //             0,0,1,0,
  //             0,0,0,1};
  //     return T;
  //   }
  // };

  // inline bool normalize(Quat& q) noexcept
  // {
  //   const float n2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
  //   if (n2 <= 0.0) return false;
  //   const float invn = 1.0 / std::sqrt(n2);
  //   q.w *= invn; q.x *= invn; q.y *= invn; q.z *= invn;
  //   return true;
  // }

  // inline bool initpos_transformation_matrix(Mat4& T_base2nworld,
  //                                           const Vec3& position,
  //                                           Quat quaternion) noexcept
  // {
  //   if (!normalize(quaternion)) return false;

  //   const float qw = quaternion.w;
  //   const float qx = quaternion.x;
  //   const float qy = quaternion.y;
  //   const float qz = quaternion.z;

  //   // yaw(gamma) from quaternion (ZYX yaw extraction)
  //   const float num = 2.0 * (qw*qz + qx*qy);
  //   const float den = 1.0 - 2.0 * (qy*qy + qz*qz);
  //   const float gamma = std::atan2(num, den);

  //   const float cg = std::cos(gamma);
  //   const float sg = std::sin(gamma);

  //   // transf_rot^T = Rz(gamma)^T = Rz(-gamma)
  //   // [ cg  sg  0 ]
  //   // [ -sg cg  0 ]
  //   // [  0   0  1 ]
  //   const float r00 =  cg, r01 =  sg, r02 = 0.0;
  //   const float r10 = -sg, r11 =  cg, r12 = 0.0;
  //   const float r20 = 0.0, r21 = 0.0, r22 = 1.0;

  //   // v = transf_rot^T * [x, y, 0]^T
  //   const float vx = r00*position.x + r01*position.y;             // =  cg*x + sg*y
  //   const float vy = r10*position.x + r11*position.y;             // = -sg*x + cg*y
  //   // vz = 0

  //   // T = [ R^T  -R^T*t ; 0 0 0 1 ]
  //   Mat4 T = Mat4::identity();
  //   T.m[0] = r00; T.m[1] = r01; T.m[2]  = r02;
  //   T.m[4] = r10; T.m[5] = r11; T.m[6]  = r12;
  //   T.m[8] = r20; T.m[9] = r21; T.m[10] = r22;

  //   T.m[3]  = -vx;
  //   T.m[7]  = -(vy);
  //   T.m[11] = -0.0;  // stays 0
  //   // last row already (0,0,0,1)

  //   T_base2nworld = T;
  //   return true;
  // }

  // Mat4 T_base2nworld = Mat4::identity();
};

#endif  // WHOLEBODY_ARC_HPP


// def initpos_transformation_matrix(self, position, quaternion):
//     qw, qx, qy, qz = self.normalize_quat(quaternion)

//     gamma = np.arctan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy**2 + qz**2))

//     cg, sg = np.cos(gamma), np.sin(gamma)

//     transf_rot = np.array([[cg, -sg, 0.0],
//                             [sg, cg, 0.0],
//                             [0.0, 0.0, 1.0]], dtype=float)

//     transf_trans = np.array([position[0], position[1], 0], dtype=float)

//     T_base2nworld = np.eye(4, dtype=float)
//     T_base2nworld[:3, :3] = transf_rot.transpose()
//     T_base2nworld[:3, 3] = -transf_rot.transpose() @ transf_trans

//     # print(T_base2nworld)
//     #
//     # print(T_base2nworld @ T_WB)

//     self.T_base2nworld = T_base2nworld


// // qw, qx, qy, qz = self.normalize_quat(self.data.qpos[3:7])

// // p = np.asarray(self.data.qpos[:3], dtype=float)
// // R = np.array([
//     [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
//     [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
//     [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
// ], dtype=float)
// T_WB = np.eye(4, dtype=float)
// T_WB[:3, :3] = R
// T_WB[:3, 3] = p

// 0.00428004, -0.0106231, 0.735388
// B2N
// 0.999601  0.000395253  0.0282255  0
// 0  0.999902  -0.0140014  0
// -0.0282282  0.0139958  0.999503  0.735388
// 0  0  0  1

// -0.0811445, 0.0568314, 0.735544
// B2N
// 0.999594  0.000281543  0.0285022  0
// 0  0.999951  -0.00987907  0
// -0.0285036  0.00987506  0.999545  0.735544
// 0  0  0  1

// -0.0929348, -0.0581182, 0.735779
// B2N
// 0.999667  0.000203133  0.0258221  0
// 8.9407e-08  0.999969  -0.00786827  0
// -0.0258229  0.00786565  0.999636  0.735779
// 0  0  0  1

// qw, qx, qy, qz = self.normalize_quat(self.data.qpos[3:7])

// p = np.asarray(self.data.qpos[:3], dtype=float)
// R = np.array([
//     [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
//     [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
//     [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
// ], dtype=float)
// T_WB = np.eye(4, dtype=float)
// T_WB[:3, :3] = R
// T_WB[:3, 3] = p
// R_rel = (self.T_base2nworld @ T_WB)[:3,:3]

// yaw_rel = np.arctan2(R_rel[1, 0], R_rel[0, 0])

// # qw, qx, qy, qz = self.normalize_quat(self.data.qpos[3:7])
// # gamma = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

// self.velocity_commands[2] = -yaw_rel*1.2  # gamma

// p_local = self.T_base2nworld @ np.hstack([p, 1.0])
// self.velocity_commands[1] = np.clip(-p_local[1], -0.2, 0.2)  # self.data.qpos[1]

// (0,0) (0,1) (0,2) (0,3)
// (1,0) (1,1) (1,2) (1,3)
// (2,0) (2,1) (2,2) (2,3)
// (3,0) (3,1) (3,2) (3,3)
