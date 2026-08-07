/******************************************************************************************
* Unitree-G1 RL Sim2Real
*
* Minimal GLFW window for the MuJoCo simulation.
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi, Taehyun Kim */

#include "g1_sim2real/mujoco_viewer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {

// The command is a trim, not a stick: a held key moves it and letting go leaves it
// where it is, so any value in between is reachable and stays put. Rates are in
// command units per second.
constexpr float kCoarseRate = 1.0f;
constexpr float kFineRate = 0.2f;

// GLFW callbacks are C function pointers with no user data, so they dispatch through
// the window's user pointer.
MujocoViewer* Self(GLFWwindow* w)
{
  return static_cast<MujocoViewer*>(glfwGetWindowUserPointer(w));
}

void KeyCallback(GLFWwindow* w, int key, int, int action, int)
{
  if (auto* self = Self(w)) self->OnKey(key, action);
}

void MouseButtonCallback(GLFWwindow* w, int, int, int)
{
  if (auto* self = Self(w)) self->OnMouseButton();
}

void CursorPosCallback(GLFWwindow* w, double x, double y)
{
  if (auto* self = Self(w)) self->OnCursorMove(x, y);
}

void ScrollCallback(GLFWwindow* w, double, double dy)
{
  if (auto* self = Self(w)) self->OnScroll(dy);
}

double WallSeconds()
{
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

// Nudge a command by `dir * rate * dt`, clamped to the normalised stick range.
float Trim(float value, float dir, float rate, double dt)
{
  return std::clamp(value + dir * rate * static_cast<float>(dt), -1.0f, 1.0f);
}

}  // namespace

MujocoViewer::MujocoViewer(const mjModel* m, const std::string& title)
: m_(m)
{
  if (!glfwInit()) throw std::runtime_error("glfwInit failed (no display?)");

  window_ = glfwCreateWindow(1280, 960, title.c_str(), nullptr, nullptr);
  if (!window_)
  {
    glfwTerminate();
    throw std::runtime_error("glfwCreateWindow failed");
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  mjv_defaultCamera(&cam_);
  mjv_defaultOption(&opt_);
  mjv_defaultScene(&scn_);
  mjr_defaultContext(&con_);

  mjv_makeScene(m_, &scn_, 2000);
  mjr_makeContext(m_, &con_, mjFONTSCALE_150);

  ResetCamera();

  glfwSetWindowUserPointer(window_, this);
  glfwSetKeyCallback(window_, KeyCallback);
  glfwSetMouseButtonCallback(window_, MouseButtonCallback);
  glfwSetCursorPosCallback(window_, CursorPosCallback);
  glfwSetScrollCallback(window_, ScrollCallback);

  std::printf("[Viewer] arrows walk/turn, A D step sideways, shift fine, X zero, "
              "SPACE pause, BACKSPACE restart, R camera, ESC quit\n");
  std::printf("[Viewer] the command latches: hold a key to move it, release to keep it\n");
}

MujocoViewer::~MujocoViewer()
{
  mjr_freeContext(&con_);
  mjv_freeScene(&scn_);

  if (window_) glfwDestroyWindow(window_);
  glfwTerminate();
}

void MujocoViewer::Track(int body_id)
{
  if (body_id < 0) return;
  cam_.type = mjCAMERA_TRACKING;
  cam_.trackbodyid = body_id;
}

void MujocoViewer::ResetCamera()
{
  cam_.distance = 3.5;
  cam_.azimuth = 120.0;
  cam_.elevation = -15.0;
}

bool MujocoViewer::ShouldClose() const
{
  return window_ == nullptr || glfwWindowShouldClose(window_);
}

bool MujocoViewer::TakeResetRequest()
{
  const bool requested = reset_requested_;
  reset_requested_ = false;
  return requested;
}

void MujocoViewer::SetCommand(float vx, float vy, float wz)
{
  vx_ = std::clamp(vx, -1.0f, 1.0f);
  vy_ = std::clamp(vy, -1.0f, 1.0f);
  wz_ = std::clamp(wz, -1.0f, 1.0f);
}

void MujocoViewer::PollCommand(double dt, float& vx, float& vy, float& wz)
{
  if (!window_) return;

  const auto held = [this](int key) { return glfwGetKey(window_, key) == GLFW_PRESS; };

  // Sign conventions match the gamepad mapping in UnitreeComm: +x forward, +y left,
  // +yaw counter-clockwise.
  // Several bindings per axis on purpose. The arrow cluster is the obvious one, but a
  // keypad arrow arrives as GLFW_KEY_KP_* when Num Lock is off, not as GLFW_KEY_DOWN,
  // and on some layouts an arrow never reaches the window at all - so W/S/Q/E work too.
  float dx = 0.0f, dy = 0.0f, dz = 0.0f;
  if (held(GLFW_KEY_UP)    || held(GLFW_KEY_W) || held(GLFW_KEY_KP_8)) dx += 1.0f;
  if (held(GLFW_KEY_DOWN)  || held(GLFW_KEY_S) || held(GLFW_KEY_KP_2)) dx -= 1.0f;
  if (held(GLFW_KEY_LEFT)  || held(GLFW_KEY_Q) || held(GLFW_KEY_KP_4)) dz += 1.0f;
  if (held(GLFW_KEY_RIGHT) || held(GLFW_KEY_E) || held(GLFW_KEY_KP_6)) dz -= 1.0f;
  if (held(GLFW_KEY_A))                                                dy += 1.0f;
  if (held(GLFW_KEY_D))                                                dy -= 1.0f;

  // Shift slows the trim down for dialling in a value precisely.
  const bool fine = held(GLFW_KEY_LEFT_SHIFT) || held(GLFW_KEY_RIGHT_SHIFT);
  const float rate = fine ? kFineRate : kCoarseRate;

  // Integrating rather than snapping means the command never steps, which matters:
  // it is part of the policy's observation, and a step there is a jolt on the robot.
  vx_ = Trim(vx_, dx, rate, dt);
  vy_ = Trim(vy_, dy, rate, dt);
  wz_ = Trim(wz_, dz, rate, dt);

  vx = vx_;
  vy = vy_;
  wz = wz_;
}

void MujocoViewer::Render(const mjModel* m, mjData* d, double sim_time,
                          const std::string& notice)
{
  if (!window_) return;

  int width = 0, height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  const mjrRect viewport = {0, 0, width, height};

  mjv_updateScene(m, d, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
  mjr_render(viewport, &scn_, &con_);

  char status[256];
  std::snprintf(status, sizeof(status),
                "t        %.2f s\nbase z   %.3f m\nvx vy wz %+.2f %+.2f %+.2f",
                sim_time, d->qpos[2], vx_, vy_, wz_);
  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, status, nullptr, &con_);

  // Top centre carries whatever the operator needs to notice right now. PAUSED and the
  // caller's notice share the slot, so compose them instead of drawing twice - two
  // overlays at the same grid position would sit on top of each other.
  std::string banner;
  if (paused_) banner = "PAUSED";
  if (!notice.empty())
  {
    if (!banner.empty()) banner += "\n";
    banner += notice;
  }
  if (!banner.empty())
  {
    // Normal size, but shadowed: it sits over the scene rather than over a panel, so it
    // needs the contrast without shouting like mjFONT_BIG does.
    mjr_overlay(mjFONT_SHADOW, mjGRID_TOP, viewport, banner.c_str(), nullptr, &con_);
  }

  // Keep the mapping on screen: this is a tool, not something worth memorising.
  mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport,
              "walk / back\nturn\nstep sideways\nfine trim\nzero\npause\nrestart\ncamera\nquit",
              "up down  or  W S\nleft right  or  Q E\nA D\nshift\nX\nspace\nbackspace\nR\nesc",
              &con_);

  glfwSwapBuffers(window_);
  glfwPollEvents();

  PaceRealtime(sim_time);
}

void MujocoViewer::PaceRealtime(double sim_time)
{
  // Physics runs far faster than real time; hold it back to 1x so the motion is
  // watchable. Re-base whenever the clock is invalidated (pause, episode restart)
  // rather than trying to make up the lost time.
  if (!clock_valid_)
  {
    clock_valid_ = true;
    sim_ref_ = sim_time;
    wall_ref_ = WallSeconds();
    return;
  }

  const double ahead = (sim_time - sim_ref_) - (WallSeconds() - wall_ref_);
  if (ahead > 0.0005)
  {
    std::this_thread::sleep_for(std::chrono::duration<double>(ahead));
  }
  else if (ahead < -1.0)
  {
    sim_ref_ = sim_time;
    wall_ref_ = WallSeconds();
  }
}

void MujocoViewer::OnKey(int key, int action)
{
  if (action != GLFW_PRESS) return;

  switch (key)
  {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window_, GLFW_TRUE);
      break;

    case GLFW_KEY_SPACE:
      paused_ = !paused_;
      clock_valid_ = false;  // do not try to catch up on the paused time
      break;

    case GLFW_KEY_BACKSPACE:
      reset_requested_ = true;
      clock_valid_ = false;
      break;

    case GLFW_KEY_X:
      vx_ = vy_ = wz_ = 0.0f;
      break;

    case GLFW_KEY_R:
      ResetCamera();
      break;

    default:
      break;
  }
}

void MujocoViewer::OnMouseButton()
{
  left_ = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  middle_ = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  right_ = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

  glfwGetCursorPos(window_, &last_x_, &last_y_);
}

void MujocoViewer::OnCursorMove(double x, double y)
{
  if (!left_ && !middle_ && !right_)
  {
    last_x_ = x;
    last_y_ = y;
    return;
  }

  const double dx = x - last_x_;
  const double dy = y - last_y_;
  last_x_ = x;
  last_y_ = y;

  int width = 0, height = 0;
  glfwGetWindowSize(window_, &width, &height);
  if (height <= 0) return;

  const bool shift = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

  mjtMouse action;
  if (right_)      action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else if (left_)  action = shift ? mjMOUSE_MOVE_H : mjMOUSE_ROTATE_V;
  else             action = mjMOUSE_ZOOM;

  mjv_moveCamera(m_, action, dx / height, dy / height, &cam_);
}

void MujocoViewer::OnScroll(double dy)
{
  mjv_moveCamera(m_, mjMOUSE_ZOOM, 0.0, -0.05 * dy, &cam_);
}
