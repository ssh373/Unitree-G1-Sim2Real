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
#include <cstdlib>
#include <algorithm>
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
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// yaml-cpp
#include <yaml-cpp/yaml.h>

// onnxruntime
#include <onnxruntime_cxx_api.h>

// g1_sim2real
#include "g1_sim2real/gamepad.hpp"
#include "g1_sim2real/utils.hpp"
#include "g1_sim2real/loop_freq_monitor.hpp"
#include "g1_sim2real/keyboard.hpp"
#include "g1_sim2real/xbox_gamepad.hpp"

// Topics
static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string HG_BMS_TOPIC = "rt/lf/bmsstate";

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

  // RL Configuration (unitree_rl_lab g1_29dof running policy)
  static const size_t NUM_ACTIONS = G1_NUM_MOTOR;                       // 29: whole-body joint position targets
  static const size_t NUM_SINGLE_OBS = 3 + 3 + 3 + 3 * NUM_ACTIONS + 2; // 98: ang_vel, gravity, cmd, q, dq, last_action, sin/cos phase
  static const size_t OBS_HISTORY_LEN = 5;                              // policy history_length (deploy.yaml)
  static const size_t NUM_OBS = NUM_SINGLE_OBS * OBS_HISTORY_LEN;       // 490

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
  ChannelSubscriberPtr<BmsState_> bms_subscriber_;

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

  /* Xbox controller (input_device: "xbox"): feeds gamepad_ in place of the wireless remote,
     so the FSM buttons and stick mapping behave identically. The wireless remote's Select is
     still honored as an emergency stop while an xbox pad is selected. */
  std::unique_ptr<g1_sim2real::XboxGamepad> xbox_;

  // Joystick command echo (see EchoJoystickCommand)
  int js_echo_cnt_ = 0;
  bool js_echo_moving_ = false;

  /* Combine mode (input_device: "combine"): keyboard and joystick together. The keyboard sets a
     held base command (cruise, e.g. vx pinned at 2.0 with 'w'), the stick adds on top of it, and
     the sum is clamped to the training range. FSM buttons work from both devices. */
  bool combine_ = false;

  /* Keyboard input, for when no wireless remote is available (input_device: "keyboard").
     The FSM triggers and the velocity command then come from the terminal instead of the sticks.
     NOTE: this means the operator has to stay at the PC and keep the terminal focused -- the
     remote is the better emergency stop whenever one is at hand. */
  std::unique_ptr<g1_sim2real::Keyboard> keyboard_;
  std::string kb_last_key_;
  std::array<float, 3> kb_cmd_ = {};   // accumulated velocity command [m/s, m/s, rad/s]
  float kb_idle_ = 0.0f;               // seconds since the last keypress
  bool kb_start_ = false;              // one-shot: Start pressed
  bool kb_policy_ = false;             // one-shot: A pressed
  static constexpr float KB_CMD_STEP = 0.1f;   // per keypress
  static constexpr float KB_DECAY_S = 1.0f;    // full scale -> 0 once idle, see keyboard_idle_timeout

  // Loop frequency monitors
  LoopFrequencyMonitor control_freq_monitor{50};  // 50 ticks @50Hz -> reports once a second
  LoopFrequencyMonitor command_writer_freq_monitor;

  // RL onnxruntime
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<Ort::AllocatedStringPtr> io_name_holders_;
  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> output_names;

  std::vector<float> input_data;
  std::array<int64_t, 2> input_shape;

  // policy (IsaacLab) joint order -> SDK motor index  (deploy.yaml: joint_ids_map)
  static constexpr std::array<int, NUM_ACTIONS> joint_ids_map = 
  {
    LeftHipPitch, RightHipPitch, WaistYaw,
    LeftHipRoll, RightHipRoll, WaistRoll,
    LeftHipYaw, RightHipYaw, WaistPitch,
    LeftKnee, RightKnee,
    LeftShoulderPitch, RightShoulderPitch,
    LeftAnklePitch, RightAnklePitch,
    LeftShoulderRoll, RightShoulderRoll,
    LeftAnkleRoll, RightAnkleRoll,
    LeftShoulderYaw, RightShoulderYaw,
    LeftElbow, RightElbow,
    LeftWristRoll, RightWristRoll,
    LeftWristPitch, RightWristPitch,
    LeftWristYaw, RightWristYaw
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

  // Tolerance on the joint position limits above. Resting against a mechanical stop is normal --
  // hang the robot and the feet dangle until the ankles sit on their limit, and MuJoCo's soft
  // joint constraint lets a joint settle ~1e-4 rad past it. Encoder calibration adds a few mrad
  // more. A joint genuinely out of range (encoder fault, broken linkage) is off by far more than
  // this, and a joint being driven into its stop shows up as torque, not position.
  static constexpr float JOINT_POS_MARGIN = 0.05f;  // [rad] ~2.9 deg

  // Upper-body joint velocity kill threshold (checked for indices >= G1_NUM_LEG_MOTOR only).
  // NOTE: the 29-DoF policy drives the arms, and sim2sim measures shoulder_pitch at 8.3 rad/s
  // while walking at 0.5 m/s and 9.0 rad/s while running, with waist_pitch touching 12.6. At
  // 7 rad/s this kill therefore fires during ordinary arm swing, not only on a runaway. Raise it
  // (15 rad/s is still less than half the motors' rated 22-37 rad/s) if that turns out to be the
  // trip on hardware.
  float joint_vel_limit = 7.f;
  
  // Kill threshold: CheckSafetyLimits drops to DAMPING when measured tau_est exceeds this.
  // Set to the G1 motor peak torques (also the training effort_limit_sim and the MuJoCo
  // <motor ctrlrange>), so crossing it means the clamp below failed or a motor is faulty --
  // not merely that the policy asked for a lot. Previously 80/120 on the legs, which sat under
  // the hardware: every normal saturation read as a fault and killed the run.
  static constexpr std::array<float, G1_NUM_MOTOR> torque_limit =
  {
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5
  };

  // Clamp threshold: LowCommandWriter caps the commanded torque at this fraction of the kill
  // threshold (0.97 -> 135 Nm on the knee, 85 Nm on the hip). The gap is what makes the kill
  // meaningful: commands stay a few Nm below it, so tau_est noise and the lag between our 500 Hz
  // estimate and the motor's own faster PD loop cannot trip it on their own.
  static constexpr float TORQUE_CLAMP_RATIO = 0.97f;

  // control state
  std::mutex cout_mutex;
  State state_ = State::WAIT_FOR_INIT_COMMAND;

  // RL_POLICY_ACTIVE
  int cnt = 0;                                       // 500Hz ticks since policy activation (gait clock)
  std::array<float, NUM_ACTIONS> rl_action_ = {};    // last raw policy output (policy joint order)

  /* Simulation-only overrides. Both are opt-in via environment variables and do nothing unless
     set, so the hardware path is unchanged. They exist because unitree_mujoco only fills
     lowstate.wireless_remote when a physical gamepad is attached: without one the FSM can never
     leave WAIT_FOR_INIT_COMMAND and the velocity command is stuck at zero.
       SIM_AUTO_START[=sec]  press Start after `sec` (default 1) and A once the default pose is held
       SIM_CMD="vx,vy,wz"    drive a fixed velocity command [m/s, m/s, rad/s] instead of the sticks  */
  bool sim_auto_start_ = false;
  float sim_auto_start_delay_ = 1.0f;
  float sim_state_time_ = 0.0f;                 // seconds spent in the current waiting state
  bool sim_cmd_active_ = false;
  std::array<float, 3> sim_cmd_ = {};
  static constexpr float SIM_CMD_RAMP_S = 2.0f; // ease the command in after the policy starts

  /* Diagnostics mirrored into the log. Without these a recording shows joint traces with no
     context: which command produced them, whether the robot was upright, and whether the torque
     clamp was doing anything. */
  std::array<float, 3> last_cmd_ = {};    // velocity command actually fed to the policy [m/s, m/s, rad/s]
  std::array<float, 2> last_phase_ = {};  // gait clock as the policy sees it (sin, cos); (0,0) while standing
  std::atomic<uint32_t> clamp_mask_{0};   // sticky bitmask, bit i set if motor i was clamped since the last log row

  // observation history: single frames, oldest -> newest (index OBS_HISTORY_LEN-1 is current)
  std::array<std::array<float, NUM_SINGLE_OBS>, OBS_HISTORY_LEN> obs_history_ = {};
  bool obs_history_initialized_ = false;

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
  void LoadSimOverrides();
  void LoadYamlConfig(const std::string& config_yaml_path);
  void PrintYamlConfig();
  void LoadOnnxModel();

  // DDS Callback functions
  void LowStateHandler(const void *message);
  void imuTorsoHandler(const void *message);
  void BmsHandler(const void *message);

  // DDS Command Writer function
  void LowCommandWriter();

  // Control functions
  void Control();
  void CheckSafetyLimits();

  // Input functions
  void UpdateKeyboardInput();
  void EchoJoystickCommand();

  // RL functions
  void ResetPolicyState();
  std::array<float, NUM_SINGLE_OBS> GetSingleObservation();
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
    std::string input_device;          // "gamepad" (wireless remote), "xbox", "keyboard" or "combine"
    std::string xbox_device;           // joystick device node for input_device: "xbox"
    float keyboard_idle_timeout;       // [s] decay the keyboard command to zero after this idle time; 0 disables
    float gait_period;                 // [s] gait clock period (deploy.yaml gait_phase.period)
    float cmd_threshold;               // [m/s] |cmd| below this -> stand, phase obs = 0
    std::array<float, 3> cmd_scale = {};
    std::array<float, 3> cmd_min = {}; // command range lower bound (vx, vy, wz)
    std::array<float, 3> cmd_max = {}; // command range upper bound; full stick = cmd_max
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

  /* Battery snapshot from rt/lf/bmsstate (published at low frequency). Logged so a recording
     shows the pack condition next to the joint traces: a policy that looks weaker than last
     week may just be running on a sagging pack. Empty columns mean the topic never arrived
     (e.g. in simulation). */
  struct BmsData
  {
    uint8_t soc = 0;        // state of charge [%]
    float voltage = 0.0f;   // pack voltage [V], sum of the reported cell voltages
    int32_t current = 0;    // pack current as reported by the BMS (negative = discharging)
    int16_t temp_max = 0;   // hottest reported sensor [C]
    uint16_t cycle = 0;     // charge cycle count
  };

  // YAML config
  YamlConfig cfg;

  // create data buffer
  g1_sim2real::DataBuffer<MotorCommand> motor_command_buffer_;
  g1_sim2real::DataBuffer<MotorState> motor_state_buffer_;
  g1_sim2real::DataBuffer<ImuState> imu_state_buffer_;
  g1_sim2real::DataBuffer<BmsData> bms_buffer_;
};

#endif  // WHOLEBODY_RL_HPP