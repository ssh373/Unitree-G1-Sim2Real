/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* MuJoCo side of the controller.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi, Taehyun Kim */

#include "g1_sim2real/mujoco_sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace g1_sim2real;

MujocoSim::MujocoSim(WholeBodyRL& controller, const std::string& model_path)
: controller_(controller)
{
  char error[1000] = "";
  m_ = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
  if (!m_) throw std::runtime_error(std::string("mj_loadXML failed: ") + error);

  if (m_->nu < G1_NUM_MOTOR)
  {
    mj_deleteModel(m_);
    throw std::runtime_error("model has " + std::to_string(m_->nu) +
                             " actuators, expected at least " +
                             std::to_string(G1_NUM_MOTOR));
  }

  d_ = mj_makeData(m_);
  if (!d_)
  {
    mj_deleteModel(m_);
    throw std::runtime_error("mj_makeData failed");
  }

  std::printf("[MujocoSim] %s\n", model_path.c_str());
  std::printf("[MujocoSim] nq=%ld nv=%ld nu=%ld timestep=%.4f\n",
              static_cast<long>(m_->nq), static_cast<long>(m_->nv),
              static_cast<long>(m_->nu), m_->opt.timestep);

  PlaceOnGround();

  // Say the schedule up front: without this the first 7 seconds look like the robot is
  // simply stuck, since the controller is only holding the default pose and the base is
  // pinned until the policy is handed control.
  std::printf("[MujocoSim] base pinned, holding the default pose for the first %.1f s;\n"
              "            the policy takes over after that\n", policy_at_);
}

MujocoSim::~MujocoSim()
{
  if (d_) mj_deleteData(d_);
  if (m_) mj_deleteModel(m_);
}

void MujocoSim::SetCommand(float lin_vel_x, float lin_vel_y, float ang_vel_z)
{
  input_.lin_vel_x = lin_vel_x;
  input_.lin_vel_y = lin_vel_y;
  input_.ang_vel_z = ang_vel_z;
}

void MujocoSim::Reset()
{
  PlaceOnGround();
  controller_.Reset();
}

void MujocoSim::PlaceOnGround()
{
  mj_resetData(m_, d_);

  // Start from the pose the policy was trained around, not the model's zero pose.
  const std::array<float, G1_NUM_MOTOR>& default_pos = controller_.default_pos();
  for (int i = 0; i < G1_NUM_MOTOR; ++i) d_->qpos[kQposOffset + i] = default_pos[i];

  d_->qpos[3] = 1.0;  // upright: identity quaternion (w, x, y, z)
  d_->qpos[4] = 0.0;
  d_->qpos[5] = 0.0;
  d_->qpos[6] = 0.0;

  // The MJCF declares the pelvis height for the all-zero pose, but the RL default has
  // the knees bent, so the feet would hang in the air. Ask MuJoCo where the ground is
  // instead of hard-coding an offset: mj_forward runs collision detection, so lower the
  // base until it reports a contact. Stays correct if the pose or the model changes.
  constexpr double kStart = 1.20;
  constexpr double kStep = 0.001;
  constexpr double kClearance = 0.002;

  d_->qpos[2] = kStart;
  bool touched = false;
  for (int k = 0; k < 1200; ++k)
  {
    mj_forward(m_, d_);
    if (d_->ncon > 0) { touched = true; break; }
    d_->qpos[2] -= kStep;
  }

  if (touched) d_->qpos[2] += kClearance;
  else std::printf("[MujocoSim] WARNING no ground contact found\n");

  for (int i = 0; i < m_->nv; ++i) d_->qvel[i] = 0.0;
  mj_forward(m_, d_);

  for (int i = 0; i < 7; ++i) pinned_base_[i] = d_->qpos[i];

  std::printf("[MujocoSim] standing at base height %.3f m\n", d_->qpos[2]);
}

void MujocoSim::PinBase()
{
  // Before the policy takes over, the controller only closes a joint-space PD loop -
  // it has no balance control, because on hardware the robot hangs from a gantry or is
  // held by the operator while it moves to the default pose. Pinning the floating base
  // reproduces that, and hands the policy the upright pose it was trained to start
  // from instead of a robot already face-down on the floor.
  for (int i = 0; i < 7; ++i) d_->qpos[i] = pinned_base_[i];
  for (int i = 0; i < 6; ++i) d_->qvel[i] = 0.0;
}

void MujocoSim::ReadState(MotorState& ms, ImuState& is) const
{
  for (int i = 0; i < G1_NUM_MOTOR; ++i)
  {
    ms.q.at(i) = static_cast<float>(d_->qpos[kQposOffset + i]);
    ms.dq.at(i) = static_cast<float>(d_->qvel[kQvelOffset + i]);
    ms.tau.at(i) = static_cast<float>(d_->qfrc_actuator[kQvelOffset + i]);
  }

  for (int i = 0; i < 4; ++i) is.quat.at(i) = static_cast<float>(d_->qpos[3 + i]);
  for (int i = 0; i < 3; ++i) is.omega.at(i) = static_cast<float>(d_->qvel[3 + i]);
}

void MujocoSim::ApplyCommand(const MotorCommand& mc, const MotorState& ms)
{
  for (int i = 0; i < G1_NUM_MOTOR; ++i)
  {
    float tau = mc.kp.at(i) * (mc.q_target.at(i) - ms.q.at(i))
              + mc.kd.at(i) * (mc.dq_target.at(i) - ms.dq.at(i))
              + mc.tau_ff.at(i);

    d_->ctrl[i] = std::clamp(tau, -torque_limit.at(i), torque_limit.at(i));
  }
}

void MujocoSim::Step()
{
  MotorState ms;
  ImuState is;
  ReadState(ms, is);

  input_.start = true;                     // no operator here: go straight for it
  input_.policy = policy_active();
  input_.damping = false;

  controller_.read(static_cast<float>(m_->opt.timestep), ms, is);
  controller_.set_input(input_);
  controller_.control();

  MotorCommand mc;
  if (controller_.write(mc)) ApplyCommand(mc, ms);

  mj_step(m_, d_);

  if (!input_.policy) PinBase();
}
