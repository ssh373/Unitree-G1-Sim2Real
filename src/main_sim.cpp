/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Runs the same controller against MuJoCo (sim2sim), no robot required.
*
*   ./wholebody_sim [config.yaml] [model.xml] [duration_s] [vx] [vy] [wz]
*
* Single-threaded: step the physics, draw a frame every 1/60 s, and let the viewer sleep
* the remainder so simulated time tracks the wall clock. Steer with the keyboard - see
* mujoco_viewer.hpp for the keys.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi, Taehyun Kim */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "g1_sim2real/mujoco_sim.hpp"
#include "g1_sim2real/mujoco_viewer.hpp"
#include "g1_sim2real/wholebody_rl.hpp"

int main(int argc, char** argv)
{
  const std::string config_path = (argc > 1) ? argv[1] : "../configs/g1_sim2real.yaml";
  const std::string model_path  = (argc > 2) ? argv[2]
                                : "../model/g1_mode15_29dof/g1_mode15_29dof.xml";
  const double duration = (argc > 3) ? std::atof(argv[3]) : 0.0;  // 0 = until closed
  const float vx0 = (argc > 4) ? std::atof(argv[4]) : 0.0f;
  const float vy0 = (argc > 5) ? std::atof(argv[5]) : 0.0f;
  const float wz0 = (argc > 6) ? std::atof(argv[6]) : 0.0f;

  try
  {
    WholeBodyRL controller(config_path);
    MujocoSim sim(controller, model_path);
    MujocoViewer viewer(sim.model());

    viewer.Track(mj_name2id(sim.model(), mjOBJ_BODY, "pelvis"));

    // Optional starting command, so a run is reproducible without touching the keyboard.
    viewer.SetCommand(vx0, vy0, wz0);

    // One display frame of simulated time per drawn frame.
    const double frame = 1.0 / 60.0;
    auto last_frame = std::chrono::steady_clock::now();

    while (!viewer.ShouldClose())
    {
      const auto now = std::chrono::steady_clock::now();
      const double wall_dt = std::chrono::duration<double>(now - last_frame).count();
      last_frame = now;

      if (viewer.TakeResetRequest()) sim.Reset();

      float vx = 0.0f, vy = 0.0f, wz = 0.0f;
      viewer.PollCommand(wall_dt, vx, vy, wz);
      sim.SetCommand(vx, vy, wz);

      if (!viewer.paused())
      {
        const double until = sim.time() + frame;
        while (sim.time() < until) sim.Step();
      }

      // Until the policy is handed control the robot just holds a pose with its base
      // pinned, which is easy to mistake for a hang. Say so on screen, and stop saying
      // it the moment the policy takes over.
      std::string notice;
      if (!sim.policy_active())
      {
        char text[128];
        std::snprintf(text, sizeof(text),
                      "holding default pose - policy starts in %.1f s",
                      sim.policy_at() - sim.time());
        notice = text;
      }

      viewer.Render(sim.model(), sim.data(), sim.time(), notice);

      if (duration > 0.0 && sim.time() >= duration) break;
    }

    const mjtNum* base = sim.data()->qpos;
    std::printf("[wholebody_sim] stopped at t=%.3f s, base x %.2f y %.2f z %.3f m\n",
                sim.time(), base[0], base[1], base[2]);
  }
  catch (const std::exception& e)
  {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
