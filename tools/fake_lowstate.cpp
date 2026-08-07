/*****************************************************************************
** Stands in for the robot so wholebody_rl can be exercised without hardware.
**
** Publishes rt/lowstate at 500 Hz with a still, upright robot (all joints at
** zero, identity orientation) and a scripted gamepad sequence, so the state
** machine walks itself through WAIT_FOR_INIT -> MOVING_TO_DEFAULT ->
** WAIT_FOR_POLICY -> RL_POLICY_ACTIVE with no one touching a controller.
**
** Run it on the loopback interface, never on the robot's network:
**   ./fake_lowstate lo
** and set `network_interface: "lo"` in configs/g1_sim2real.yaml.
**
** This validates the software path only - DDS transport, observation
** assembly, ONNX inference, command writing. It says nothing about whether
** the controller behaves correctly on a real robot.
*****************************************************************************/
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <thread>

#include "g1_sim2real/gamepad.hpp"
#include "g1_sim2real/utils.hpp"

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

namespace {

constexpr int    NUM_MOTOR    = 35;    // LowState_ carries 35 motor slots
constexpr int    PERIOD_US    = 2000;  // 500 Hz, same rate as the real robot
constexpr uint8_t MODE_MACHINE = 5;    // G1 model number

// The controller can take several seconds to come up (DDS discovery plus the
// motion-switcher timeout when no robot answers), so a single button press is
// easy to miss. Tap each button every PRESS_EVERY seconds instead: the state
// machine ignores a button that is not meaningful in its current state, so
// repeating is harmless and makes the run independent of startup timing.
constexpr double PRESS_EVERY = 4.0;
constexpr double PRESS_HOLD  = 0.30;

bool Tapping(double t)
{
  return std::fmod(t, PRESS_EVERY) < PRESS_HOLD;
}

}  // namespace

int main(int argc, char** argv)
{
  const std::string interface = (argc > 1) ? argv[1] : "lo";
  // Tap `start` until this time, then tap `A`. Leave enough room for the 5 s
  // move-to-default to finish before the policy is requested.
  const double hand_over_at = (argc > 2) ? std::atof(argv[2]) : 20.0;
  if (interface != "lo")
  {
    std::printf("[WARN] publishing fake robot state on '%s', not loopback.\n"
                "       Only do this on an isolated network.\n", interface.c_str());
  }

  ChannelFactory::Instance()->Init(0, interface);

  ChannelPublisher<LowState_> publisher("rt/lowstate");
  publisher.InitChannel();

  std::printf("[fake_lowstate] publishing rt/lowstate on %s at %d Hz\n",
              interface.c_str(), 1000000 / PERIOD_US);
  std::printf("[fake_lowstate] tapping 'start' every %.0fs until t=%.0fs, then 'A'\n",
              PRESS_EVERY, hand_over_at);

  uint32_t tick = 0;
  const auto t0 = std::chrono::steady_clock::now();

  while (true)
  {
    const double t = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    LowState_ low_state{};
    low_state.mode_machine() = MODE_MACHINE;
    low_state.mode_pr() = 0;  // PR mode, matches Mode::PR
    low_state.tick() = tick++;

    // A still robot: every joint parked at zero, no motion, no load.
    for (int i = 0; i < NUM_MOTOR; ++i)
    {
      low_state.motor_state()[i].mode() = 1;
      low_state.motor_state()[i].q() = 0.0f;
      low_state.motor_state()[i].dq() = 0.0f;
      low_state.motor_state()[i].ddq() = 0.0f;
      low_state.motor_state()[i].tau_est() = 0.0f;
    }

    // Upright: identity quaternion, no angular rate, gravity on -z.
    low_state.imu_state().quaternion() = {1.0f, 0.0f, 0.0f, 0.0f};
    low_state.imu_state().gyroscope() = {0.0f, 0.0f, 0.0f};
    low_state.imu_state().accelerometer() = {0.0f, 0.0f, -9.81f};
    low_state.imu_state().rpy() = {0.0f, 0.0f, 0.0f};

    // Scripted gamepad. The layout must match REMOTE_DATA_RX in gamepad.hpp,
    // which wholebody_rl memcpy's straight out of these 40 bytes.
    REMOTE_DATA_RX rx{};
    std::memset(rx.buff, 0, sizeof(rx.buff));
    const bool tap = Tapping(t);
    rx.RF_RX.btn.components.start = (tap && t < hand_over_at) ? 1 : 0;
    rx.RF_RX.btn.components.A     = (tap && t >= hand_over_at) ? 1 : 0;
    rx.RF_RX.lx = 0.0f;   // sticks centred: the policy gets a zero command
    rx.RF_RX.ly = 0.0f;
    rx.RF_RX.rx = 0.0f;
    rx.RF_RX.ry = 0.0f;
    std::memcpy(&low_state.wireless_remote()[0], rx.buff, 40);

    // wholebody_rl drops any frame whose CRC does not check out.
    low_state.crc() =
        g1_sim2real::Crc32Core((uint32_t*)&low_state, (sizeof(LowState_) >> 2) - 1);

    publisher.Write(low_state);

    std::this_thread::sleep_for(std::chrono::microseconds(PERIOD_US));
  }

  return 0;
}
