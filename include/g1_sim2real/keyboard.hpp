/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Non-blocking keyboard reader, for driving the FSM when no wireless remote is available.
* Adapted from unitree_rl_lab (deploy/include/isaaclab/devices/keyboard/keyboard.h).
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

#pragma once
#ifndef G1_SIM2REAL_KEYBOARD_HPP
#define G1_SIM2REAL_KEYBOARD_HPP

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <termios.h>
#include <unistd.h>

namespace g1_sim2real {

/**
 * Puts the terminal into raw mode and keeps the most recent keypress available to other threads.
 * key() returns "" once ~80 ms pass with nothing pressed, so callers can edge-detect a press by
 * comparing against the previous value.
 *
 * The controller exits through std::exit(), which does not run destructors of locals, so the
 * terminal restore is also registered with atexit(): without it a kill would leave the shell
 * without echo or line editing.
 */
class Keyboard
{
 public:
  Keyboard()
  {
    tcgetattr(fileno(stdin), &original_);
    settings_ = original_;

    saved() = original_;
    static const bool restore_registered = []{ std::atexit(&Keyboard::Restore); return true; }();
    (void)restore_registered;

    settings_.c_lflag &= (~ICANON & ~ECHO);
    tcsetattr(fileno(stdin), TCSANOW, &settings_);

    running_ = true;
    thread_ = std::thread([this]{ while (running_) Read(); });
  }

  ~Keyboard()
  {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    Restore();
  }

  std::string key() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return key_;
  }

  static void Restore() { tcsetattr(fileno(stdin), TCSANOW, &saved()); }

 private:
  static termios& saved() { static termios s{}; return s; }

  void Read()
  {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fileno(stdin), &fds);

    timeval tv{0, 80000};  // 80 ms: long enough to coalesce key repeat, short enough to feel live

    std::string k;
    if (select(fileno(stdin) + 1, &fds, nullptr, nullptr, &tv) > 0)
    {
      char c = '\0';
      if (read(fileno(stdin), &c, 1) == 1 && c != '\033') k = std::string(1, c);
      // escape sequences (arrow keys) are drained and ignored
      else if (c == '\033') { char seq[2]; ssize_t n = read(fileno(stdin), seq, 2); (void)n; }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    key_ = k;
  }

  std::atomic<bool> running_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  std::string key_;
  termios original_{}, settings_{};
};

}  // namespace g1_sim2real

#endif  // G1_SIM2REAL_KEYBOARD_HPP
