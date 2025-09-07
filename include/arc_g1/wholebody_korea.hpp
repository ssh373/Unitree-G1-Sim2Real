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

#ifndef WHOLEBODY_KOREA_HPP
#define WHOLEBODY_KOREA_HPP

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

// onnxruntime
#include <onnxruntime_cxx_api.h>

// Topics
static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

// namespace
using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace std;

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
  double time_, time_abs;
  double control_dt_;  // [2ms]
  double duration_;    // [3 s]
  int counter_;
  Mode mode_pr_;
  uint8_t mode_machine_;
  std::atomic<bool> safe_freq_ = false;

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
  static constexpr std::array<int, 29> isaaclab2mujoco = 
  {
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
  };

  // mujoco2isaaclab
  static constexpr std::array<int, 29> mujoco2isaaclab = 
  {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
  };

  static constexpr std::array<float, 29> joint_pos_min = 
  {
    -2.5307, -0.5236, -2.7576, -0.30, -0.80, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.30, -0.80, -0.2618,
    -0.6, -0.6, -0.6,
    -3.0892, -1.5882, -1.618, -1.0472, -1.97222, -1.61443, -0.61443,
    -3.0892, -2.2515, -1.618, -1.0472, -1.97222, -1.61443, -0.61443
  };

  static constexpr std::array<float, 29> joint_pos_max = 
  {
    2.8798, 2.9671, 2.7576, 2.5449, 0.51, 0.2618, 
    2.8798, 0.5236, 2.7576, 2.5449, 0.51, 0.2618,
    0.6, 0.6, 0.6,
    2.6704, 2.2515, 1.618, 2.0944, 1.97222, 1.61443, 0.61443,
    2.6704, 1.5882, 1.618, 2.0944, 1.97222, 1.61443, 0.61443
  };

  static constexpr std::array<float, 29> torque_limit = 
  {
    100, 100, 100, 100, 40, 30,
    100, 100, 100, 100, 40, 30,
    300, 300, 300,
    60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60
  };

  // control state
  std::mutex cout_mutex;
  State state_ = State::WAIT_FOR_INIT_COMMAND;

  // RL policy
  int cnt = 0;
  float period = 4.0; // 0.8  1.2
  size_t start_idx = 9;
  static constexpr size_t num_obs = 98;
  static constexpr size_t num_action = 29;
  std::vector<float> input_tensor_values_;
  std::array<int64_t, 2> input_shape_ = {1, 98};
  std::array<float, num_action> rl_action_ = {};

  std::array<float, G1_NUM_MOTOR> new_default_pos = {};
  // 5.7488e-01, -1.4046e-01,  1.0274e-03, -2.1038e-04, -5.4188e-01, 1.0358e-01,
  // -5.2558e-01,  7.8469e-02,  8.3221e-02,  4.9121e-01, 7.5621e-02, -2.9229e-02,
  // -1.3908e-02, -1.3627e-02, -1.1454e-01,
  // -2.5529e-01,  2.0068e-01, -3.3256e-01,  5.5452e-01,-2.9860e-16, -5.1404e-19, -1.1986e-19,
  // 5.2707e-01, -1.9586e-02, -3.0400e-01, 1.1011e+00,  3.3827e-16, -3.0702e-19,  3.1789e-19

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
  void LoadOnnxModel(const std::string& model_path);
  void imuTorsoHandler(const void *message);
  void LowStateHandler(const void *message);
  void LowCommandWriter();
  std::array<float, 3> GetGravityOrientation(const std::array<float, 4>& q);
  std::vector<float> GetMockObservation();
  std::vector<float> GetObservation();
  std::array<float, 29> RunInference();
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
  arc_g1::DataBuffer<RLAction> rl_action_buffer_;
  YamlConfig cfg;
};

#endif  // WHOLEBODY_KOREA_HPP
