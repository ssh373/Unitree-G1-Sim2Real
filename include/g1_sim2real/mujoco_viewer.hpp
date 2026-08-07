/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Minimal GLFW window for the MuJoCo simulation.
*
* Deliberately small: it owns a window, a camera and the keyboard/mouse, and nothing
* else. Rendering itself lives in libmujoco (mjv_* / mjr_*), which is the same code
* MuJoCo's own `simulate` app draws with - we just skip its 11k lines of UI, and with
* them the lodepng dependency and the two-thread synchronisation it forces.
*
* Keyboard doubles as the gamepad: on the real robot the velocity command comes from
* the sticks inside LowState_, here it comes from held keys.
*
*   up / down      walk forward / back     (vx)   also W / S, keypad 8 / 2
*   left / right   turn left / right       (wz)   also Q / E, keypad 4 / 6
*   A / D          step left / right       (vy)
*   shift          hold for a slower trim
*   X              zero the command
*   SPACE          pause
*   BACKSPACE      restart the episode
*   R              reset the camera
*   ESC            quit
*   drag / shift+drag / scroll             orbit / pan / zoom
*
* Arrows steer the way they would in any game - forward and turn - because that is what
* an operator reaches for first. Sideways walking is the rare one, so it gets its own
* keys rather than stealing left/right.
*
* The command behaves as a trim, not a spring-loaded stick: a held key moves it and
* releasing leaves it there, so 0.75 is just as reachable as 1.0. X returns it to zero.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

#ifndef G1_SIM2REAL_MUJOCO_VIEWER_HPP
#define G1_SIM2REAL_MUJOCO_VIEWER_HPP

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <string>

class MujocoViewer
{
 public:
  MujocoViewer(const mjModel* m, const std::string& title = "G1 Sim2Sim");
  ~MujocoViewer();

  MujocoViewer(const MujocoViewer&) = delete;
  MujocoViewer& operator=(const MujocoViewer&) = delete;

  // Follow a body so the robot does not walk out of frame.
  void Track(int body_id);

  bool ShouldClose() const;
  bool paused() const { return paused_; }

  // True once per BACKSPACE press; clears itself.
  bool TakeResetRequest();

  // Held keys trim a latched, normalised command - the same -1..1 the gamepad produces
  // on the robot. Call once per frame with the elapsed wall time.
  void PollCommand(double dt, float& vx, float& vy, float& wz);

  // Seed the latched command, e.g. from the command line, so a run can start already
  // walking. Keys trim it from there.
  void SetCommand(float vx, float vy, float wz);

  // Draw one frame, then sleep so simulated time tracks wall-clock time. Pass the
  // simulated time so pacing survives a pause or an episode restart. `notice` is drawn
  // centred at the top while it is non-empty, for things the operator needs to know
  // only for a while - such as the policy not being in charge yet.
  void Render(const mjModel* m, mjData* d, double sim_time,
              const std::string& notice = {});

  // Called by the GLFW trampolines.
  void OnKey(int key, int action);
  void OnMouseButton();
  void OnCursorMove(double x, double y);
  void OnScroll(double dy);

 private:
  void ResetCamera();
  void PaceRealtime(double sim_time);

  const mjModel* m_ = nullptr;
  GLFWwindow* window_ = nullptr;

  mjvCamera cam_;
  mjvOption opt_;
  mjvScene scn_;
  mjrContext con_;

  bool paused_ = false;
  bool reset_requested_ = false;

  // latched command; keys trim it, X zeroes it
  float vx_ = 0.0f, vy_ = 0.0f, wz_ = 0.0f;

  // mouse
  bool left_ = false, middle_ = false, right_ = false;
  double last_x_ = 0.0, last_y_ = 0.0;

  // realtime pacing
  bool clock_valid_ = false;
  double sim_ref_ = 0.0;
  double wall_ref_ = 0.0;
};

#endif  // G1_SIM2REAL_MUJOCO_VIEWER_HPP
