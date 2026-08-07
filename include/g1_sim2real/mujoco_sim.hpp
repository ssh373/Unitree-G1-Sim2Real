/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* MuJoCo side of the controller: what UnitreeComm is to the real robot, this is to the
* simulator. It owns mjModel/mjData and translates between them and the controller's
* vocabulary in robot_types.hpp.
*
* Two things differ from the robot and both live here:
*
*   - the MJCF actuators are <motor>, i.e. torque in, so the PD loop the Unitree motor
*     driver closes on its own is closed here instead
*   - there is no IMU sensor in the model, so base orientation and angular rate are read
*     off the free joint: qpos[3..6] is the base quaternion, qvel[3..5] the body-frame
*     angular velocity
*
* The joint ordering in model/g1_mode15_29dof matches the DDS motor indices, so no
* remapping is needed between the two sides.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

#ifndef G1_SIM2REAL_MUJOCO_SIM_HPP
#define G1_SIM2REAL_MUJOCO_SIM_HPP

#include <mujoco/mujoco.h>

#include <string>

#include "g1_sim2real/robot_types.hpp"
#include "g1_sim2real/wholebody_rl.hpp"

class MujocoSim
{
 public:
  MujocoSim(WholeBodyRL& controller, const std::string& model_path);
  ~MujocoSim();

  MujocoSim(const MujocoSim&) = delete;
  MujocoSim& operator=(const MujocoSim&) = delete;

  // One physics step: sensors in, controller, torques out, mj_step. The caller owns
  // the loop.
  void Step();

  // Velocity command, normalised -1..1 like the gamepad sticks on the robot.
  void SetCommand(float lin_vel_x, float lin_vel_y, float ang_vel_z);

  // Back to the start of an episode: default pose on the ground, controller reset.
  void Reset();

  double time() const { return d_->time; }
  double policy_at() const { return policy_at_; }
  bool policy_active() const { return time() >= policy_at_; }

  const mjModel* model() const { return m_; }
  mjData* data() { return d_; }

 private:
  void PlaceOnGround();
  void PinBase();
  void ReadState(g1_sim2real::MotorState& ms, g1_sim2real::ImuState& is) const;
  void ApplyCommand(const g1_sim2real::MotorCommand& mc,
                    const g1_sim2real::MotorState& ms);

  WholeBodyRL& controller_;

  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;

  // qpos/qvel offsets of the first actuated joint, past the 7/6 of the free joint
  static constexpr int kQposOffset = 7;
  static constexpr int kQvelOffset = 6;

  // Simulated seconds of holding the default pose before the policy takes over. Until
  // then the base is pinned; see PinBase().
  double policy_at_ = 7.0;

  mjtNum pinned_base_[7] = {};
  g1_sim2real::CommandInput input_;
};

#endif  // G1_SIM2REAL_MUJOCO_SIM_HPP
