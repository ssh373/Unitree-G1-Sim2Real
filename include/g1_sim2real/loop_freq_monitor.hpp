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

/* Authors: Sol Choi */

#pragma once
#ifndef LOOP_FREQ_MONITOR
#define LOOP_FREQ_MONITOR

#include <atomic>
#include <chrono>

class LoopFrequencyMonitor 
{
  public:
    LoopFrequencyMonitor(int check_interval = 500)
    : interval_(check_interval), tick_count_(0)
    {
      last_check_time_ = std::chrono::high_resolution_clock::now();
    }
    ~LoopFrequencyMonitor() = default;

    void Tick() 
    {
      using namespace std::chrono;
      if (++tick_count_ % interval_ == 0) 
      {
        auto now = high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_check_time_);
        double hz = static_cast<double>(interval_) / elapsed.count();
        frequency_.store(hz);
        last_check_time_ = now;
      }
    }

    double GetFrequency() const 
    {
      return frequency_.load();
    }

  private:
    int interval_;
    int tick_count_;
    std::atomic<double> frequency_{0.0};
    std::chrono::high_resolution_clock::time_point last_check_time_;
};

#endif  // LOOP_FREQ_MONITOR
