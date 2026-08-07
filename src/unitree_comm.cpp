/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Real-robot comm layer: everything that speaks DDS to the G1.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi, Taehyun Kim */

#include "g1_sim2real/unitree_comm.hpp"

#include <algorithm>
#include <cstring>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

namespace {
// Thread periods, in microseconds
constexpr int COMMAND_WRITER_PERIOD_US = g1_sim2real::CONTROL_DT * 1000000;  // 500Hz
constexpr int CONTROL_LOOP_PERIOD_US = g1_sim2real::COMMAND_DT * 1000000;    // 50Hz
}  // namespace

UnitreeComm::UnitreeComm(WholeBodyRL& controller, const std::string& network_interface)
: controller_(controller)
{
  InitUnitreeChannel(network_interface);
  InitPublisher();
  InitSubscriber();
  InitThread();
}

UnitreeComm::~UnitreeComm()
{
  should_exit_ = true;
  if (monitor_thread_.joinable()) monitor_thread_.join();
}

void UnitreeComm::Run()
{
  while (!controller_.should_exit() && !should_exit_)
  {
    std::this_thread::sleep_for(std::chrono::microseconds(2000));  // 2ms
  }
  should_exit_ = true;
}


/*****************************************************************************
** Initialize functions
*****************************************************************************/
void UnitreeComm::InitUnitreeChannel(const std::string& network_interface)
{
  // initialize unitree channel
  ChannelFactory::Instance()->Init(0, network_interface);

  // motion switcher client
  msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
  msc_->SetTimeout(5.0f);
  msc_->Init();

  std::string form, name;
  while (msc_->CheckMode(form, name), !name.empty())
  {
    if (msc_->ReleaseMode()) std::cout << "Failed to switch to Release Mode\n";
    sleep(5);
  }
}

void UnitreeComm::InitPublisher()
{
  // create publisher
  lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
  lowcmd_publisher_->InitChannel();
}

void UnitreeComm::InitSubscriber()
{
  // create subscriber
  lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
  lowstate_subscriber_->InitChannel(std::bind(&UnitreeComm::LowStateHandler, this, std::placeholders::_1), 1);

  imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
  imutorso_subscriber_->InitChannel(std::bind(&UnitreeComm::imuTorsoHandler, this, std::placeholders::_1), 1);
}

void UnitreeComm::InitThread()
{
  // create threads
  command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, COMMAND_WRITER_PERIOD_US, &UnitreeComm::LowCommandWriter, this); // 500Hz

  monitor_thread_ = std::thread(&UnitreeComm::MonitorThread, this);
}

void UnitreeComm::MonitorThread()
{
  const float control_dt_ = g1_sim2real::CONTROL_DT;
  const float command_dt_ = g1_sim2real::COMMAND_DT;

  while (!should_exit_)
  {
    // control freq
    double control_hz = control_freq_monitor.GetFrequency();

    // command writer freq
    double command_writer_hz = command_writer_freq_monitor.GetFrequency();

    if ((1 / command_dt_ - 0.01) <= control_hz && control_hz <= (1 / command_dt_ + 0.01) && !safe_freq_ &&
        (1 / control_dt_ - 0.01) <= command_writer_hz && command_writer_hz <= (1 / control_dt_ + 0.01))
    {
      safe_freq_ = true;
      std::cout << "\033[32m[Freq Monitor]\033[0m Control freq stable!"
                << " (Control loop: " << control_hz << " Hz, " << "Command Writer loop: " << command_writer_hz << " Hz)" << std::endl;
    }
    else if ((1 / command_dt_ - 0.01) >= control_hz && control_hz >= (1 / command_dt_ + 0.01) &&
             (1 / control_dt_ - 0.01) >= command_writer_hz && command_writer_hz >= (1 / control_dt_ + 0.01))
      safe_freq_ = false;

    if (!safe_freq_)
    {
      std::lock_guard<std::mutex> lock(cout_mutex_);
      std::cout << "\033[31m[Freq Monitor]\033[0m Control loop: " << control_hz << " Hz, " << "Command Writer loop: " << command_writer_hz << " Hz" << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // 1s
  }
}


/*****************************************************************************
** DDS Callback functions
*****************************************************************************/
void UnitreeComm::LowStateHandler(const void *message)
{
  LowState_ low_state = *(const LowState_ *)message;
  if (low_state.crc() != g1_sim2real::Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1))
  {
    std::cout << "[ERROR] CRC Error" << std::endl;
    return;
  }

  // get motor state
  g1_sim2real::MotorState ms_tmp;
  for (int i = 0; i < g1_sim2real::G1_NUM_MOTOR; ++i)
  {
    ms_tmp.q.at(i) = low_state.motor_state()[i].q();
    ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    ms_tmp.tau.at(i) = low_state.motor_state()[i].tau_est();
  }

  // get imu state
  g1_sim2real::ImuState imu_tmp;
  imu_tmp.omega = low_state.imu_state().gyroscope();
  imu_tmp.quat = low_state.imu_state().quaternion();

  // hand the sensors to the controller
  controller_.read(g1_sim2real::CONTROL_DT, ms_tmp, imu_tmp);
  last_motor_state_.SetData(ms_tmp);

  // update gamepad
  memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
  gamepad_.update(rx_.RF_RX);

  // Map the remote onto the controller's operator input. The sign flips are a
  // property of this gamepad, so they belong here rather than in the observation.
  g1_sim2real::CommandInput in;
  in.start = static_cast<int>(gamepad_.start.pressed) == 1;
  in.policy = static_cast<int>(gamepad_.A.pressed) == 1;
  in.damping = static_cast<int>(gamepad_.select.pressed) == 1;
  in.lin_vel_x = static_cast<float>(gamepad_.ly);
  in.lin_vel_y = static_cast<float>(gamepad_.lx) * -1.0f;
  in.ang_vel_z = static_cast<float>(gamepad_.rx) * -1.0f;
  controller_.set_input(in);

  // update mode machine
  if (mode_machine_ != low_state.mode_machine())
  {
    if (mode_machine_ == 0) std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
    mode_machine_ = low_state.mode_machine();
  }

  if (static_cast<int>(gamepad_.select.pressed) == 1) should_exit_ = true;

  // report robot status every second
  if (++counter_ % 500 == 0)
  {
    counter_ = 0;
  }
}

void UnitreeComm::imuTorsoHandler(const void *message)
{
  IMUState_ imu_torso = *(const IMUState_ *)message;
  auto &rpy = imu_torso.rpy();
  // if (counter_ % 500 == 0)
  //   printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
  (void)rpy;
}


/*****************************************************************************
** Recurrent loops
*****************************************************************************/
void UnitreeComm::LowCommandWriter()
{
  command_writer_freq_monitor.Tick();

  // The controller decimates internally, so mirror that here to report the rate
  // the policy is actually running at.
  if (writer_step_ % g1_sim2real::DECIMATION == 0) control_freq_monitor.Tick();
  ++writer_step_;

  // one controller step at 500Hz (the policy itself decimates to 50Hz)
  controller_.control();

  g1_sim2real::MotorCommand mc;
  if (!controller_.write(mc))
  {
    if (controller_.should_exit()) should_exit_ = true;
    return;
  }

  LowCmd_ dds_low_command;
  dds_low_command.mode_pr() = static_cast<uint8_t>(controller_.mode_pr());
  dds_low_command.mode_machine() = mode_machine_;

  for (size_t i = 0; i < g1_sim2real::G1_NUM_MOTOR; i++)
  {
    dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
    dds_low_command.motor_cmd().at(i).tau()  = mc.tau_ff.at(i);
    dds_low_command.motor_cmd().at(i).q()    = mc.q_target.at(i);
    dds_low_command.motor_cmd().at(i).dq()   = mc.dq_target.at(i);
    dds_low_command.motor_cmd().at(i).kp()   = mc.kp.at(i);
    dds_low_command.motor_cmd().at(i).kd()   = mc.kd.at(i);
  }

  // Torque limit. When the PD command would exceed the joint's limit we hand the
  // motor a pure torque instead: this is a property of the Unitree motor driver,
  // which is why it stays here rather than in the controller.
  const std::shared_ptr<const g1_sim2real::MotorState> ms = last_motor_state_.GetData();
  if (ms)
  {
    for (size_t i = 0; i < g1_sim2real::NUM_ACTIONS; i++)
    {
      float torque_des = controller_.rl_kp(i) * (mc.q_target.at(i) - ms->q.at(i))
                       + controller_.rl_kd(i) * (0 - ms->dq.at(i));

      if (std::abs(torque_des) > g1_sim2real::torque_limit.at(i))  // torque command
      {
        torque_des = std::clamp(torque_des, -g1_sim2real::torque_limit.at(i), g1_sim2real::torque_limit.at(i));
        dds_low_command.motor_cmd().at(i).tau() = torque_des;
        dds_low_command.motor_cmd().at(i).q()   = 0;
        dds_low_command.motor_cmd().at(i).dq()  = 0;
        dds_low_command.motor_cmd().at(i).kp()  = 0.00001;
        dds_low_command.motor_cmd().at(i).kd()  = 0.00001;
      }
    }
  }

  dds_low_command.crc() = g1_sim2real::Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
  lowcmd_publisher_->Write(dds_low_command);

  if (controller_.should_exit()) should_exit_ = true;
}
