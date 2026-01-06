/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Deploy a trained RL locomotion policy on Unitree G1 hardware
*
*     https://github.com/S-CHOI-S/Unitree-G1-Sim2Real.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi, Taehyun Kim */

#ifndef WHOLEBODY_RL_HPP
#define WHOLEBODY_RL_HPP

#include <cmath>
#include <vector>
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <string>
#include <functional>
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>

// DDS includes
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL includes
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// yaml-cpp
#include <yaml-cpp/yaml.h>

// onnxruntime
#include <onnxruntime_cxx_api.h>

// g1_sim2real
#include "g1_sim2real/gamepad.hpp"
#include "g1_sim2real/utils.hpp"
#include "g1_sim2real/loop_freq_monitor.hpp"

// Topics
static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

// namespace
using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class WholeBodyRL {
 public:
  WholeBodyRL(const std::string& model_path);
  ~WholeBodyRL();

  /*****************************************************************************
  ** Define variables
  *****************************************************************************/
  // Robot Configuration
  static const int G1_NUM_MOTOR = 29;
  static const int G1_NUM_LEG_MOTOR = 12;
  static const int G1_NUM_UPPER_MOTOR = G1_NUM_MOTOR - G1_NUM_LEG_MOTOR;

  // RL Configuration
  static const size_t NUM_OBS = 47;
  static const size_t NUM_ACTIONS = 12;

  // Control Frequencies (in seconds)
  static constexpr float CONTROL_DT = 0.002f;  // 500Hz
  static constexpr float COMMAND_DT = 0.02f;   // 50Hz
  
  // Thread Timing Configuration (in microseconds)
  static constexpr int COMMAND_WRITER_PERIOD_US = CONTROL_DT * 1000000;  // 500Hz (0.002s)
  static constexpr int CONTROL_LOOP_PERIOD_US = COMMAND_DT * 1000000;    // 50Hz (0.02s)

  std::atomic<bool> should_exit_ = false;
  std::atomic<bool> logging_active_ = false;

  /*****************************************************************************
  ** Define functions
  *****************************************************************************/

 private:
  /*****************************************************************************
  ** Define publisher & subscriber & thread
  *****************************************************************************/
  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;
  
  // Robot state publisher
  ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;

  // Robot state subscriber
  ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;

  // Threads
  ThreadPtr command_writer_ptr_, control_thread_ptr_;
  std::thread monitor_thread_;
  std::thread logger_thread_;

  /*****************************************************************************
  ** Define enum class
  *****************************************************************************/
  // Robot Control States
  enum class State 
  {
    WAIT_FOR_INIT_COMMAND,
    MOVING_TO_DEFAULT,
    WAIT_FOR_POLICY_COMMAND,
    RL_POLICY_ACTIVE,
    DAMPING_STATE
  };
  
  // Robot Control Modes (PR/AB for ankle joints)
  enum class Mode 
  {
    PR = 0,  // Series Control for Pitch/Roll Joints
    AB = 1   // Parallel Control for A/B Joints
  };

  // Robot Joint Indices
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
  uint8_t mode_machine_;      // robot model
  Mode mode_pr_;              // control mode for ankle joints
  float time_, time_abs;      // time trackers
  float control_dt_;          // [2ms]: 500hz
  float command_dt_;          // [20ms]: 50hz
  float duration_;            // [5 s]: duration to move to default pose
  int counter_;               // control loop counter (terminal info)
  std::atomic<bool> safe_freq_ = false;

  // Gamepad (joystick)
  Gamepad gamepad_;
  REMOTE_DATA_RX rx_;

  // Loop frequency monitors
  LoopFrequencyMonitor control_freq_monitor;
  LoopFrequencyMonitor command_writer_freq_monitor;

  // RL onnxruntime
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> output_names;

  std::vector<float> input_data;
  std::array<int64_t, 2> input_shape;

  // isaaclab2mujoco
  static constexpr std::array<int, NUM_ACTIONS> isaaclab2mujoco = 
  {
    0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11
  };

  // mujoco2isaaclab
  static constexpr std::array<int, NUM_ACTIONS> mujoco2isaaclab = 
  {
    LeftHipPitch, RightHipPitch, LeftHipRoll, RightHipRoll, LeftHipYaw, RightHipYaw,
    LeftKnee, RightKnee, LeftAnklePitch, RightAnklePitch, LeftAnkleRoll, RightAnkleRoll
  };

  // joint limits & torque limits
  static constexpr std::array<float, G1_NUM_MOTOR> joint_pos_min = 
  {
    -2.5307, -0.5236, -2.7576, -0.15, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.15, -0.87267, -0.2618,
    -0.75, -0.75, -0.75,
    -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
    -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443
  };

  static constexpr std::array<float, G1_NUM_MOTOR> joint_pos_max = 
  {
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    0.75, 0.75, 0.75,
    2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
    2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443
  };

  float joint_vel_limit = 7.f;
  
  static constexpr std::array<float, G1_NUM_MOTOR> torque_limit = 
  {
    80, 120, 80, 120, 50, 50,
    80, 120, 80, 120, 50, 50,
    80, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5
  };

  // control state
  std::mutex cout_mutex;
  State state_ = State::WAIT_FOR_INIT_COMMAND;

  // RL_POLICY_ACTIVE
  int cnt = 0;
  float stride_a = 8.0e-7;
  float stride_b = 1.0;
  float eps = 1e-07;
  size_t start_idx = 9;
  std::array<float, NUM_ACTIONS> rl_action_ = {};

  /*****************************************************************************
  ** Define functions
  *****************************************************************************/
  // Initialize functions
  void InitUnitreeChannel();
  void InitPublisher();
  void InitSubscriber();
  void InitThread();
  
  // Thread functions
  void MonitorThread();
  void LoggerThread();

  // Config & Model loading functions
  void LoadYamlConfig(const std::string& config_yaml_path);
  void PrintYamlConfig();
  void LoadOnnxModel();

  // DDS Callback functions
  void LowStateHandler(const void *message);
  void imuTorsoHandler(const void *message);

  // DDS Command Writer function
  void LowCommandWriter();

  // Control functions
  void Control();
  void CheckSafetyLimits();

  // RL functions
  std::vector<float> GetObservation();
  std::array<float, NUM_ACTIONS> RunInference();

  // Helper functions
  std::array<float, 3> GetGravityOrientation(const std::array<float, 4>& q);
  std::array<float, 3> Quat2RPY(const std::array<float, 4>& q);

  /*****************************************************************************
  ** Define structure & data buffer
  *****************************************************************************/
  struct YamlConfig
  {
    std::string networkInterface;
    std::string policy_path;
    std::string log_dir;
    std::string log_file;
    std::array<float, G1_NUM_MOTOR> default_pos = {};
    std::array<float, G1_NUM_MOTOR> rl_kp = {};
    std::array<float, G1_NUM_MOTOR> rl_kd = {};
    std::array<float, G1_NUM_MOTOR> init_kp = {};
    std::array<float, G1_NUM_MOTOR> init_kd = {};
    float ang_vel_scale;
    float dof_pos_scale;
    float dof_vel_scale;
    float action_scale;
    std::array<float, 3> cmd_scale = {};
    std::array<float, 3> max_cmd = {};
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

  struct ImuState
  {
    std::array<float, 3> rpy = {};
    std::array<float, 3> omega = {};
    std::array<float, 4> quat = {};
  };

  // YAML config
  YamlConfig cfg;

  // create data buffer
  g1_sim2real::DataBuffer<MotorCommand> motor_command_buffer_;
  g1_sim2real::DataBuffer<MotorState> motor_state_buffer_;
  g1_sim2real::DataBuffer<ImuState> imu_state_buffer_;
};

#endif  // WHOLEBODY_RL_HPP