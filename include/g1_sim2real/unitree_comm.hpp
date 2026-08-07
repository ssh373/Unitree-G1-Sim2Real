/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Real-robot comm layer: everything that speaks DDS to the G1.
*
* Owns the loops. The controller is a pure read() -> control() -> write() step, so the
* 500 Hz / 50 Hz cadence, CRC, mode_machine and the gamepad all live here and nowhere else.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

#ifndef G1_SIM2REAL_UNITREE_COMM_HPP
#define G1_SIM2REAL_UNITREE_COMM_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include "g1_sim2real/gamepad.hpp"
#include "g1_sim2real/loop_freq_monitor.hpp"
#include "g1_sim2real/robot_types.hpp"
#include "g1_sim2real/wholebody_rl.hpp"

// Topics
static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

class UnitreeComm
{
 public:
  UnitreeComm(WholeBodyRL& controller, const std::string& network_interface);
  ~UnitreeComm();

  // Blocks until the controller asks to exit.
  void Run();

 private:
  void InitUnitreeChannel(const std::string& network_interface);
  void InitPublisher();
  void InitSubscriber();
  void InitThread();

  // DDS callbacks
  void LowStateHandler(const void* message);
  void imuTorsoHandler(const void* message);

  // Recurrent loop: drives the controller at 500Hz, which decimates to 50Hz itself
  void LowCommandWriter();
  void MonitorThread();

  WholeBodyRL& controller_;

  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;

  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imutorso_subscriber_;

  unitree::common::ThreadPtr command_writer_ptr_;
  std::thread monitor_thread_;
  size_t writer_step_ = 0;

  // Gamepad lives on the robot: it arrives inside LowState_.wireless_remote
  unitree::common::Gamepad gamepad_;
  unitree::common::REMOTE_DATA_RX rx_;

  LoopFrequencyMonitor control_freq_monitor;
  LoopFrequencyMonitor command_writer_freq_monitor;

  std::atomic<bool> should_exit_ = false;
  std::atomic<bool> safe_freq_ = false;
  std::mutex cout_mutex_;

  uint8_t mode_machine_ = 0;
  int counter_ = 0;

  // last state received, used for the driver-side torque clamp
  g1_sim2real::DataBuffer<g1_sim2real::MotorState> last_motor_state_;
};

#endif  // G1_SIM2REAL_UNITREE_COMM_HPP
