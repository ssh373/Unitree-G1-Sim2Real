/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Types shared between the controller and whatever drives it.
*
* Nothing here knows about DDS or MuJoCo: this is the whole vocabulary the control
* code and its driver need to agree on. UnitreeComm / MujocoSim fill MotorState /
* ImuState / CommandInput and consume MotorCommand.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

#ifndef G1_SIM2REAL_ROBOT_TYPES_HPP
#define G1_SIM2REAL_ROBOT_TYPES_HPP

#include <array>
#include <string>

namespace g1_sim2real {

// Robot configuration
constexpr int G1_NUM_MOTOR = 29;
constexpr int G1_NUM_LEG_MOTOR = 12;

// RL configuration
constexpr size_t NUM_OBS = 47;
constexpr size_t NUM_ACTIONS = 12;

// Control frequencies (in seconds)
constexpr float CONTROL_DT = 0.002f;  // 500Hz, PD / command write
constexpr float COMMAND_DT = 0.02f;   // 50Hz, policy inference
constexpr size_t DECIMATION = 10;     // COMMAND_DT / CONTROL_DT

// Ankle joint control mode
enum class Mode
{
  PR = 0,  // Series control for Pitch/Roll joints
  AB = 1   // Parallel control for A/B joints
};

// Robot joint indices. The MuJoCo model (model/g1_mode15_29dof) uses the same
// ordering, so no remapping is needed between the two backends.
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

struct MotorState
{
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
  std::array<float, G1_NUM_MOTOR> tau = {};
};

struct ImuState
{
  std::array<float, 3> omega = {};   // body-frame angular velocity
  std::array<float, 4> quat = {};    // w, x, y, z
};

struct MotorCommand
{
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

// What the operator asks for, however it was obtained: gamepad on the
// real robot, keyboard or a script in simulation.
struct CommandInput
{
  bool start = false;    // move to the default pose
  bool policy = false;   // hand over to the RL policy
  bool damping = false;  // kill: fall back to damping

  // Normalised stick, -1..1, scaled by cmd_scale * max_cmd into the observation.
  float lin_vel_x = 0.0f;
  float lin_vel_y = 0.0f;
  float ang_vel_z = 0.0f;
};

// Per-joint limits, checked by the controller either way.
constexpr std::array<float, G1_NUM_MOTOR> joint_pos_min =
{
  -2.5307, -0.5236, -2.7576, -0.15, -0.87267, -0.2618,
  -2.5307, -2.9671, -2.7576, -0.15, -0.87267, -0.2618,
  -0.75, -0.75, -0.75,
  -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
  -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443
};

constexpr std::array<float, G1_NUM_MOTOR> joint_pos_max =
{
  2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
  2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
  0.75, 0.75, 0.75,
  2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
  2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443
};

constexpr std::array<float, G1_NUM_MOTOR> torque_limit =
{
  80, 120, 80, 120, 50, 50,
  80, 120, 80, 120, 50, 50,
  80, 50, 50,
  25, 25, 25, 25, 25, 5, 5,
  25, 25, 25, 25, 25, 5, 5
};

constexpr float joint_vel_limit = 7.0f;

}  // namespace g1_sim2real

#endif  // G1_SIM2REAL_ROBOT_TYPES_HPP
