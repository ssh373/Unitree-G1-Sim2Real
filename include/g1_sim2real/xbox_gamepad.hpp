/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Xbox controller reader (Linux joystick API), for driving the FSM with a USB/bluetooth
* Xbox pad instead of the robot's wireless remote (input_device: "xbox").
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

#pragma once
#ifndef G1_SIM2REAL_XBOX_GAMEPAD_HPP
#define G1_SIM2REAL_XBOX_GAMEPAD_HPP

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <fcntl.h>
#include <linux/joystick.h>
#include <sys/select.h>
#include <unistd.h>

#include "g1_sim2real/gamepad.hpp"

namespace g1_sim2real {

/**
 * Reads an Xbox controller from /dev/input/jsX and republishes it as the Unitree remote's
 * xRockerBtnDataStruct, so the existing Gamepad class (smoothing, dead zone, edge detection)
 * and every FSM check work unchanged whichever device is selected.
 *
 * Axis/button numbers follow the kernel xpad driver (wired / USB dongle). A pad connected
 * through xpadneo (bluetooth) can enumerate differently; verify with `jstest` before use.
 *
 * On disconnect the state is zeroed (sticks centered, buttons released), which coasts the
 * velocity command to zero but leaves the policy running; the wireless remote's Select is
 * still honored as an emergency stop by the controller.
 */
class XboxGamepad
{
 public:
  explicit XboxGamepad(const std::string& device) : device_(device)
  {
    std::memset(&data_, 0, sizeof(data_));
    running_ = true;
    thread_ = std::thread([this]{ Loop(); });
  }

  ~XboxGamepad()
  {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) close(fd_);
  }

  unitree::common::xRockerBtnDataStruct data() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_;
  }

  bool connected() const { return connected_; }

 private:
  // xpad axes
  static constexpr int AX_LX = 0;  // left stick horizontal, right = +32767
  static constexpr int AX_LY = 1;  // left stick vertical, down = +32767
  static constexpr int AX_LT = 2;  // left trigger, released = -32767
  static constexpr int AX_RX = 3;  // right stick horizontal
  static constexpr int AX_RY = 4;  // right stick vertical
  static constexpr int AX_RT = 5;  // right trigger
  static constexpr int AX_DPAD_X = 6;
  static constexpr int AX_DPAD_Y = 7;

  // xpad buttons (XB_ prefix: linux/input.h already defines BTN_A, BTN_START, ...)
  static constexpr int XB_A = 0;
  static constexpr int XB_B = 1;
  static constexpr int XB_X = 2;
  static constexpr int XB_Y = 3;
  static constexpr int XB_LB = 4;
  static constexpr int XB_RB = 5;
  static constexpr int XB_BACK = 6;   // -> select
  static constexpr int XB_START = 7;  // -> start

  void Loop()
  {
    while (running_)
    {
      if (fd_ < 0 && !Open())
      {
        // retry once a second without spamming the terminal
        for (int i = 0; i < 10 && running_; ++i) usleep(100000);
        continue;
      }

      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(fd_, &fds);
      timeval tv{0, 80000};

      if (select(fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;

      js_event ev;
      const ssize_t n = read(fd_, &ev, sizeof(ev));
      if (n != static_cast<ssize_t>(sizeof(ev)))
      {
        Disconnect();
        continue;
      }
      Apply(ev);
    }
  }

  bool Open()
  {
    fd_ = open(device_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ < 0) return false;

    char name[128] = "unknown";
    ioctl(fd_, JSIOCGNAME(sizeof(name)), name);
    printf("[INPUT] Xbox controller connected: %s (%s)\n", name, device_.c_str());
    connected_ = true;
    return true;
  }

  void Disconnect()
  {
    close(fd_);
    fd_ = -1;
    connected_ = false;
    printf("\033[33m[INPUT] Xbox controller disconnected -- command zeroed, reconnecting...\033[0m\n");
    std::lock_guard<std::mutex> lock(mutex_);
    std::memset(&data_, 0, sizeof(data_));
  }

  void Apply(const js_event& ev)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& b = data_.btn.components;

    switch (ev.type & ~JS_EVENT_INIT)
    {
      case JS_EVENT_AXIS:
      {
        const float v = ev.value / 32767.0f;
        switch (ev.number)
        {
          // remap to the Unitree remote's frame: ly/ry up = +, lx/rx right = +
          case AX_LX: data_.lx = v; break;
          case AX_LY: data_.ly = -v; break;
          case AX_RX: data_.rx = v; break;
          case AX_RY: data_.ry = -v; break;
          case AX_LT: data_.L2 = (v + 1.0f) * 0.5f;
                      b.L2 = data_.L2 > 0.5f; break;
          case AX_RT: b.R2 = (v + 1.0f) * 0.5f > 0.5f; break;
          case AX_DPAD_X: b.left = ev.value < 0; b.right = ev.value > 0; break;
          case AX_DPAD_Y: b.up = ev.value < 0; b.down = ev.value > 0; break;
          default: break;
        }
        break;
      }
      case JS_EVENT_BUTTON:
      {
        const bool on = ev.value != 0;
        switch (ev.number)
        {
          case XB_A: b.A = on; break;
          case XB_B: b.B = on; break;
          case XB_X: b.X = on; break;
          case XB_Y: b.Y = on; break;
          case XB_LB: b.L1 = on; break;
          case XB_RB: b.R1 = on; break;
          case XB_BACK: b.select = on; break;
          case XB_START: b.start = on; break;
          default: break;
        }
        break;
      }
      default: break;
    }
  }

  std::string device_;
  int fd_ = -1;
  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  unitree::common::xRockerBtnDataStruct data_;
};

}  // namespace g1_sim2real

#endif  // G1_SIM2REAL_XBOX_GAMEPAD_HPP
