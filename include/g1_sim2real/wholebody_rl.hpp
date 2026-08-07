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

/*
 * Control only. Nothing in here talks to a robot or to a simulator.
 *
 * A comm/sim layer drives it as
 *
 *     read(dt, motor_state, imu_state);   // sensors in
 *     set_input(command_input);           // operator in
 *     control();                          // one 500 Hz step (inference every 10th)
 *     write(motor_command);               // q_target / kp / kd / tau_ff out
 *
 * and owns the loop. See unitree_comm.hpp (real robot, DDS) and mujoco_sim.hpp (MuJoCo).
 */

#ifndef WHOLEBODY_RL_HPP
#define WHOLEBODY_RL_HPP

#include <cmath>
#include <vector>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <array>
#include <string>
#include <functional>
#include <iostream>
#include <fstream>
#include <thread>
#include <filesystem>

// yaml-cpp
#include <yaml-cpp/yaml.h>

// onnxruntime
#include <onnxruntime_cxx_api.h>

// 1.12 replaced Session::GetInputName/GetOutputName with the *NameAllocated
// variants used below. Fail here rather than deep inside a template error.
#if ORT_API_VERSION < 12
#error "onnxruntime 1.12 or newer is required (set ONNXRUNTIME_VERSION accordingly)"
#endif

// g1_sim2real
#include "g1_sim2real/robot_types.hpp"
#include "g1_sim2real/utils.hpp"

using namespace g1_sim2real;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class WholeBodyRL {
 public:
  explicit WholeBodyRL(const std::string& config_yaml_path);
  ~WholeBodyRL();

  /*****************************************************************************
  ** Comm / sim seam
  *****************************************************************************/
  // Sensors in. dt is the control period the caller is actually running at.
  void read(float dt, const MotorState& ms, const ImuState& is);

  // Operator in: buttons and velocity command, however the caller got them.
  void set_input(const CommandInput& in);

  // One control step. Call at 1/CONTROL_DT (500 Hz); the policy runs every
  // DECIMATION-th call (50 Hz).
  void control();

  // Command out. Returns false until the first control() has produced one.
  bool write(MotorCommand& mc);

  // Restart the episode: back to WAIT_FOR_INIT_COMMAND with every counter cleared.
  // Called when the world being driven is reset or reloaded.
  void Reset();

  bool should_exit() const { return should_exit_; }
  Mode mode_pr() const { return mode_pr_; }
  const std::string& network_interface() const { return cfg.networkInterface; }
  const std::array<float, G1_NUM_MOTOR>& default_pos() const { return cfg.default_pos; }
  float rl_kp(size_t i) const { return cfg.rl_kp.at(i); }
  float rl_kd(size_t i) const { return cfg.rl_kd.at(i); }

  /*****************************************************************************
  ** Define variables
  *****************************************************************************/
  // G1_NUM_MOTOR, NUM_OBS, NUM_ACTIONS, CONTROL_DT, COMMAND_DT, the joint indices
  // and the joint/torque limits all come from robot_types.hpp, which UnitreeComm and
  // MujocoSim share, so both sides speak the same vocabulary.

  std::atomic<bool> should_exit_ = false;
  std::atomic<bool> logging_active_ = false;

 private:
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

  /*****************************************************************************
  ** Define variables
  *****************************************************************************/
  Mode mode_pr_;              // control mode for ankle joints
  float time_, time_abs;      // time trackers
  float control_dt_;          // [2ms]: 500hz
  float command_dt_;          // [20ms]: 50hz
  float duration_;            // [5 s]: duration to move to default pose

  // Operator input, refreshed through set_input()
  CommandInput input_;

  std::thread logger_thread_;

  // RL onnxruntime
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Own the io name strings; input_names/output_names point into these
  std::string input_name_holder_;
  std::string output_name_holder_;

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

  // control state
  State state_ = State::WAIT_FOR_INIT_COMMAND;

  // RL_POLICY_ACTIVE
  int cnt = 0;         // 500Hz tick, drives the gait phase
  size_t step_cnt = 0; // 500Hz tick, decimates the policy to 50Hz
  size_t start_idx = 9;
  std::array<float, NUM_ACTIONS> rl_action_ = {};

  /*****************************************************************************
  ** Define functions
  *****************************************************************************/
  // Thread functions
  void LoggerThread();

  // Config & Model loading functions
  void LoadYamlConfig(const std::string& config_yaml_path);
  void PrintYamlConfig();
  void LoadOnnxModel();

  // Control functions
  void Control();
  void CheckSafetyLimits();

  // RL functions
  std::vector<float> GetObservation();
  std::array<float, NUM_ACTIONS> RunInference();

  // Helper functions
  std::array<float, 3> GetGravityOrientation(const std::array<float, 4>& q);

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

  // YAML config
  YamlConfig cfg;

  // create data buffer
  g1_sim2real::DataBuffer<MotorCommand> motor_command_buffer_;
  g1_sim2real::DataBuffer<MotorState> motor_state_buffer_;
  g1_sim2real::DataBuffer<ImuState> imu_state_buffer_;
};

#endif  // WHOLEBODY_RL_HPP
