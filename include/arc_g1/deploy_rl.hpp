#ifndef DEPLOY_RL_HPP
#define DEPLOY_RL_HPP

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

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <algorithm>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <onnxruntime_cxx_api.h>
// #include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include <onnxruntime/core/providers/onnxruntime_provider_factory.h>

constexpr float Pi = 3.141592654f;
constexpr float Pi_2 = 1.57079632f;

class RLController 
{
  public:
    RLController(const std::string& interface = "eth0");
    ~RLController();

    /*****************************************************************************
    ** Define functions
    *****************************************************************************/
    void Initialize();
    void MoveArmsToTarget();
    void ResetArms();
    void Shutdown();
    void LoadOnnxModel(const std::string& model_path);
    std::vector<float> InferenceOnCpu(const std::vector<float>& input_data);
    void InferenceOnGpu(const std::string& model_path);
    void ControlWithPolicy();

  private:
    /*****************************************************************************
    ** Define publisher & subscriber
    *****************************************************************************/
    unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> arm_sdk_publisher_;
    unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> low_state_subscriber_;
    unitree_hg::msg::dds_::LowCmd_ msg_;
    unitree_hg::msg::dds_::LowState_ state_msg_;

    /*****************************************************************************
    ** Define joint index
    *****************************************************************************/
    enum class JointIndex 
    {
      // Left leg
      LeftHipPitch, LeftHipRoll, LeftHipYaw, LeftKnee,
      LeftAnklePitch, LeftAnkleRoll,

      // Right leg
      RightHipPitch, RightHipRoll, RightHipYaw, RightKnee,
      RightAnklePitch, RightAnkleRoll,

      // Waist
      WaistYaw, WaistRoll, WaistPitch,

      // Left arm
      LeftShoulderPitch, LeftShoulderRoll, LeftShoulderYaw,
      LeftElbow, LeftWristRoll, LeftWristPitch, LeftWristYaw,

      // Right arm
      RightShoulderPitch, RightShoulderRoll, RightShoulderYaw,
      RightElbow, RightWristRoll, RightWristPitch, RightWristYaw,

      // Not used
      NotUsedJoint, NotUsedJoint1, NotUsedJoint2, NotUsedJoint3,
      NotUsedJoint4, NotUsedJoint5
    };

    /*****************************************************************************
    ** Define variables
    *****************************************************************************/
    std::array<JointIndex, 29> g1_29dof_joints_;
    std::array<float, 29> init_pos_;
    std::array<float, 29> target_pos_;
    std::array<float, 29> current_jpos_;

    std::array<float, 29> joint_kps_;
    std::array<float, 29> joint_kds_;

    float kp_, kd_, dq_, tau_ff_;
    float weight_, weight_rate_;
    float control_dt_, max_joint_velocity_;
    float delta_weight_, max_joint_delta_;
    std::chrono::milliseconds sleep_time_;

    // onnxruntime::Env env_;
    // onnxruntime::InferenceSession* onnx_session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;


    /*****************************************************************************
    ** Define functions
    *****************************************************************************/
    void InitPublisher();
    void InitSubscriber();
    void SleepCycle();
    void WaitForUser(const std::string& prompt);
    std::array<float, 3> GetGravityOrientation(const std::array<float, 4>& q);
};

#endif // DEPLOY_RL_HPP
