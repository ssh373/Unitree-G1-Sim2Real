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

/* Authors: Sol Choi, Taehyun Kim */

#include "g1_sim2real/wholebody_rl.hpp"

#include <sstream>

WholeBodyRL::WholeBodyRL(const std::string& policy_path)
: time_(0.0),
  time_abs(0.0),
  control_dt_(CONTROL_DT), // 500Hz
  command_dt_(COMMAND_DT), // 50Hz
  duration_(5.0), // move to default pose
  counter_(0),
  mode_pr_(Mode::PR),
  mode_machine_(0)
{
  LoadSimOverrides();
  LoadYamlConfig(policy_path);
  if (cfg.input_device == "keyboard")
  {
    keyboard_ = std::make_unique<g1_sim2real::Keyboard>();
    std::cout << "\033[33m[INPUT] Keyboard control: keep this terminal focused. The wireless remote\n"
              << "        is the safer stop whenever one is available.\033[0m" << std::endl;
  }
  else if (cfg.input_device == "xbox")
  {
    xbox_ = std::make_unique<g1_sim2real::XboxGamepad>(cfg.xbox_device);
    std::cout << "\033[33m[INPUT] Xbox controller (" << cfg.xbox_device << "): Start stand, A run policy,\n"
              << "        Back stop. The wireless remote's Select still works as an emergency stop.\033[0m" << std::endl;
  }
  else if (cfg.input_device == "combine")
  {
    combine_ = true;
    keyboard_ = std::make_unique<g1_sim2real::Keyboard>();
    xbox_ = std::make_unique<g1_sim2real::XboxGamepad>(cfg.xbox_device);
    std::cout << "\033[33m[INPUT] Combine: keyboard holds a base command (w/s/a/d/q/e, space clears),\n"
              << "        the joystick adds on top. Buttons work from both (1/Start, 2/A, x/Back).\n"
              << "        Without an xbox pad the wireless remote's sticks are used instead.\033[0m" << std::endl;
  }
  LoadOnnxModel();
  InitUnitreeChannel();
  InitPublisher();
  InitSubscriber();
  InitThread();
}

WholeBodyRL::~WholeBodyRL()
{
  // should_exit_ = true;
  logging_active_ = false;

  if (monitor_thread_.joinable()) monitor_thread_.join();
  if (logger_thread_.joinable()) logger_thread_.join();
}


/*****************************************************************************
** Initialize functions
*****************************************************************************/
void WholeBodyRL::InitUnitreeChannel()
{
  // initialize unitree channel
  ChannelFactory::Instance()->Init(0, cfg.networkInterface);

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

void WholeBodyRL::InitPublisher()
{
  // create publisher
  lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
  lowcmd_publisher_->InitChannel();
}

void WholeBodyRL::InitSubscriber()
{
  // create subscriber
  lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
  lowstate_subscriber_->InitChannel(std::bind(&WholeBodyRL::LowStateHandler, this, std::placeholders::_1), 1);

  imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
  imutorso_subscriber_->InitChannel(std::bind(&WholeBodyRL::imuTorsoHandler, this, std::placeholders::_1), 1);
}


/*****************************************************************************
** Thread functions
*****************************************************************************/
void WholeBodyRL::InitThread()
{
  // create threads
  command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, COMMAND_WRITER_PERIOD_US, &WholeBodyRL::LowCommandWriter, this); // 500Hz
  control_thread_ptr_ = CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, CONTROL_LOOP_PERIOD_US, &WholeBodyRL::Control, this); // 50Hz

  monitor_thread_ = std::thread(&WholeBodyRL::MonitorThread, this);
  logger_thread_ = std::thread(&WholeBodyRL::LoggerThread, this); 
}

void WholeBodyRL::MonitorThread() 
{
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
      std::lock_guard<std::mutex> lock(cout_mutex);
      std::cout << "\033[31m[Freq Monitor]\033[0m Control loop: " << control_hz << " Hz, " << "Command Writer loop: " << command_writer_hz << " Hz" << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // 1s
  }
}

void WholeBodyRL::LoggerThread()
{
  std::ofstream log_file(cfg.log_file);

  if (!log_file.is_open()) 
  {
    std::cerr << "[Logger] Failed to open log.csv" << std::endl;
    return;
  }

  log_file << "time";
  for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << ",q" << i;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << ",qdes" << i;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << ",qdot" << i;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << ",tau" << i;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << ",taudes" << i;
  // context for the joint traces above: what was asked of the robot, how it was oriented, and
  // which of the safety mechanisms were engaged
  log_file << ",cmd_x,cmd_y,cmd_w"          // velocity command fed to the policy
           << ",phase_sin,phase_cos"        // gait clock; (0,0) means the policy is in standing mode
           << ",roll,pitch,yaw"             // pelvis IMU orientation
           << ",gyro_x,gyro_y,gyro_z"       // pelvis IMU angular rate
           << ",fsm_state"                  // 0 wait_init, 1 moving, 2 wait_policy, 3 rl_active, 4 damping
           << ",clamp_mask";                // bit i set if motor i hit the torque clamp since the last row
  log_file << "\n";
  log_file.flush();

  while (!should_exit_) 
  {
    const auto ms = motor_state_buffer_.GetData();
    const auto mc = motor_command_buffer_.GetData();

    if (ms && mc && logging_active_) 
    {
      log_file << time_abs;
      for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << "," << ms->q.at(i);
      for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << "," << mc->q_target.at(i);
      for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << "," << ms->dq.at(i);
      for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << "," << ms->tau.at(i);
      for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << "," << mc->kp.at(i) * (mc->q_target.at(i) - ms->q.at(i)) + mc->kd.at(i) * (mc->dq_target.at(i) - ms->dq.at(i));

      const auto is = imu_state_buffer_.GetData();
      log_file << "," << last_cmd_[0] << "," << last_cmd_[1] << "," << last_cmd_[2]
               << "," << last_phase_[0] << "," << last_phase_[1];
      if (is) log_file << "," << is->rpy[0]   << "," << is->rpy[1]   << "," << is->rpy[2]
                       << "," << is->omega[0] << "," << is->omega[1] << "," << is->omega[2];
      else    log_file << ",,,,,,";
      log_file << "," << static_cast<int>(state_)
               << "," << clamp_mask_.exchange(0, std::memory_order_relaxed);
      log_file << "\n";
      log_file.flush();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  log_file.close();
  std::cout << "[Logger] Stopped logging." << std::endl;
}


/*****************************************************************************
** Config & Model loading functions
*****************************************************************************/
void WholeBodyRL::LoadSimOverrides()
{
  // Opt-in simulation helpers; absent env vars leave every hardware behaviour untouched.
  if (const char* v = std::getenv("SIM_AUTO_START"))
  {
    const std::string val(v);
    sim_auto_start_ = (val != "0");
    if (sim_auto_start_)
    {
      try { if (!val.empty() && val != "1") sim_auto_start_delay_ = std::stof(val); } catch (...) {}
      std::cout << "\033[33m[SIM] SIM_AUTO_START: Start in " << sim_auto_start_delay_
                << " s, A once the default pose is held. DO NOT set this on hardware.\033[0m" << std::endl;
    }
  }

  if (const char* v = std::getenv("SIM_CMD"))
  {
    std::stringstream ss(v);
    std::string tok;
    size_t i = 0;
    while (i < sim_cmd_.size() && std::getline(ss, tok, ',')) 
    {
      try { sim_cmd_[i++] = std::stof(tok); } catch (...) { break; }
    }
    sim_cmd_active_ = (i == sim_cmd_.size());
    if (sim_cmd_active_)
      std::cout << "\033[33m[SIM] SIM_CMD: velocity command fixed at [" << sim_cmd_[0] << ", "
                << sim_cmd_[1] << ", " << sim_cmd_[2] << "], eased in over " << SIM_CMD_RAMP_S
                << " s. DO NOT set this on hardware.\033[0m" << std::endl;
    else
      std::cerr << "[SIM] SIM_CMD must be \"vx,vy,wz\" -- ignored" << std::endl;
  }
}

void WholeBodyRL::LoadYamlConfig(const std::string& config_yaml_path)
{
  try
  {
    // load yaml file
    if (!std::filesystem::exists(config_yaml_path))
      throw std::runtime_error("Config file not found: " + config_yaml_path);
    
    YAML::Node config = YAML::LoadFile(config_yaml_path);
    
    if (!config["network_interface"])
      throw std::runtime_error("Missing required field: network_interface");
    
    // network interface
    cfg.networkInterface = config["network_interface"].as<std::string>();
    
    // onnx path
    cfg.policy_path = config["policy_path"].as<std::string>();
    
    // log directory
    cfg.log_dir = config["log_dir"].as<std::string>();
    
    // get current time for log file name
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm{};
    localtime_r(&t, &tm);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    const std::filesystem::path dir = "../logs/" + cfg.log_dir;
    std::filesystem::create_directories(dir);
    
    cfg.log_file = (dir / ("log_" + oss.str() + ".csv")).string();
    
    // default pos
    std::vector<float> default_angles = config["default_angles"].as<std::vector<float>>();
    std::vector<float> arm_waist_target = config["arm_waist_target"].as<std::vector<float>>();
    if (default_angles.size() + arm_waist_target.size() != G1_NUM_MOTOR)
      throw std::runtime_error("YAML default_angles + arm_waist_target size mismatch");
    std::copy(default_angles.begin(), default_angles.end(), cfg.default_pos.begin());
    std::copy(arm_waist_target.begin(), arm_waist_target.end(), cfg.default_pos.begin() + default_angles.size());
    
    // init kp
    std::vector<float> init_leg_kps = config["init_leg_kps"].as<std::vector<float>>();
    std::vector<float> init_arm_waist_kps = config["init_arm_waist_kps"].as<std::vector<float>>();
    if (init_leg_kps.size() + init_arm_waist_kps.size() != G1_NUM_MOTOR)
      throw std::runtime_error("YAML init_leg_kps + init_arm_waist_kps size mismatch");
    std::copy(init_leg_kps.begin(), init_leg_kps.end(), cfg.init_kp.begin());
    std::copy(init_arm_waist_kps.begin(), init_arm_waist_kps.end(), cfg.init_kp.begin() + init_leg_kps.size());
    
    // init kd
    std::vector<float> init_leg_kds = config["init_leg_kds"].as<std::vector<float>>();
    std::vector<float> init_arm_waist_kds = config["init_arm_waist_kds"].as<std::vector<float>>();
    if (init_leg_kds.size() + init_arm_waist_kds.size() != G1_NUM_MOTOR)
      throw std::runtime_error("YAML init_leg_kds + init_arm_waist_kds size mismatch");
    std::copy(init_leg_kds.begin(), init_leg_kds.end(), cfg.init_kd.begin());
    std::copy(init_arm_waist_kds.begin(), init_arm_waist_kds.end(), cfg.init_kd.begin() + init_leg_kds.size());
    
    // RL kp
    std::vector<float> kps = config["kps"].as<std::vector<float>>();
    std::vector<float> arm_kps = config["arm_waist_kps"].as<std::vector<float>>();
    if (kps.size() + arm_kps.size() != G1_NUM_MOTOR)
      throw std::runtime_error("YAML kps + arm_waist_kps size mismatch");
    std::copy(kps.begin(), kps.end(), cfg.rl_kp.begin());
    std::copy(arm_kps.begin(), arm_kps.end(), cfg.rl_kp.begin() + kps.size());
    
    // RL kd
    std::vector<float> kds = config["kds"].as<std::vector<float>>();
    std::vector<float> arm_kds = config["arm_waist_kds"].as<std::vector<float>>();
    if (kds.size() + arm_kds.size() != G1_NUM_MOTOR)
      throw std::runtime_error("YAML kds + arm_waist_kds size mismatch");
    std::copy(kds.begin(), kds.end(), cfg.rl_kd.begin());
    std::copy(arm_kds.begin(), arm_kds.end(), cfg.rl_kd.begin() + kds.size());
    
    cfg.ang_vel_scale = config["ang_vel_scale"].as<float>();
    cfg.dof_pos_scale = config["dof_pos_scale"].as<float>();
    cfg.dof_vel_scale = config["dof_vel_scale"].as<float>();
    cfg.action_scale = config["action_scale"].as<float>();
    cfg.input_device = config["input_device"] ? config["input_device"].as<std::string>() : "gamepad";
    if (cfg.input_device != "gamepad" && cfg.input_device != "keyboard" && cfg.input_device != "xbox"
        && cfg.input_device != "combine")
      throw std::runtime_error("input_device must be \"gamepad\", \"keyboard\", \"xbox\" or \"combine\"");
    cfg.xbox_device = config["xbox_device"] ? config["xbox_device"].as<std::string>() : "/dev/input/js0";
    cfg.keyboard_idle_timeout = config["keyboard_idle_timeout"] ? config["keyboard_idle_timeout"].as<float>() : 0.0f;

    cfg.gait_period = config["gait_period"].as<float>();
    cfg.cmd_threshold = config["cmd_threshold"].as<float>();
    if (cfg.gait_period <= 0.f)
      throw std::runtime_error("YAML gait_period must be > 0");

    auto load_vec3 = [&](const char* key, std::array<float, 3>& dst)
    {
      std::vector<float> v = config[key].as<std::vector<float>>();
      if (v.size() != 3)
        throw std::runtime_error(std::string("YAML ") + key + " must have 3 elements");
      std::copy(v.begin(), v.end(), dst.begin());
    };
    load_vec3("cmd_scale", cfg.cmd_scale);
    load_vec3("cmd_min", cfg.cmd_min);
    load_vec3("cmd_max", cfg.cmd_max);
    for (int i = 0; i < 3; ++i)
      if (cfg.cmd_min[i] > cfg.cmd_max[i])
        throw std::runtime_error("YAML cmd_min must be <= cmd_max");
    
    PrintYamlConfig();
  }
  catch (const std::exception& e)
  {
    throw std::runtime_error("Failed to load config: " + std::string(e.what()));
  }
}

void WholeBodyRL::PrintYamlConfig()
{
  std::cout << "\n============================== [RL Config Loaded] ==============================\n";
  std::cout << "  Model path       : " << cfg.policy_path << "\n";

  std::cout << "  Scale factors:\n";
  std::cout << "    ang_vel_scale  : " << cfg.ang_vel_scale << "\n";
  std::cout << "    dof_pos_scale  : " << cfg.dof_pos_scale << "\n";
  std::cout << "    dof_vel_scale  : " << cfg.dof_vel_scale << "\n";
  std::cout << "    action_scale   : " << cfg.action_scale << "\n";
  std::cout << "    cmd_scale      : [" << cfg.cmd_scale[0] << ", " << cfg.cmd_scale[1] << ", " << cfg.cmd_scale[2] << "]\n";
  std::cout << "    cmd_min        : [" << cfg.cmd_min[0] << ", " << cfg.cmd_min[1] << ", " << cfg.cmd_min[2] << "]\n";
  std::cout << "    cmd_max        : [" << cfg.cmd_max[0] << ", " << cfg.cmd_max[1] << ", " << cfg.cmd_max[2] << "]\n";
  std::cout << "  Input device     : " << cfg.input_device;
  if (cfg.input_device == "keyboard")
    std::cout << "  (1 stand, 2 run policy, wasdqe drive, space stop, x kill"
              << (cfg.keyboard_idle_timeout > 0.f ? ", auto-stop after " + std::to_string(cfg.keyboard_idle_timeout) + "s idle)" : ")");
  else if (cfg.input_device == "xbox")
    std::cout << "  (" << cfg.xbox_device << ": Start stand, A run policy, Back stop, sticks drive)";
  else if (cfg.input_device == "combine")
    std::cout << "  (keyboard base command + joystick on top; buttons from both)";
  std::cout << "\n";
  std::cout << "  Gait clock:\n";
  std::cout << "    gait_period    : " << cfg.gait_period << "\n";
  std::cout << "    cmd_threshold  : " << cfg.cmd_threshold << "\n";

  std::cout << "  Model dimensions:\n";
  std::cout << "    num_single_obs : " << NUM_SINGLE_OBS << "\n";
  std::cout << "    history_len    : " << OBS_HISTORY_LEN << "\n";
  std::cout << "    num_obs        : " << NUM_OBS << "\n";
  std::cout << "    num_actions    : " << NUM_ACTIONS << "\n";
  std::cout << "================================================================================\n" << std::endl;
}

void WholeBodyRL::LoadOnnxModel()
{
  // initialize onnx runtime env
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_cpu_RL");

  // initialize input data
  input_data = std::vector<float>(NUM_OBS, 0.0f);
  input_shape = {1, NUM_OBS};

  // session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // load onnx model
  session_ = std::make_unique<Ort::Session>(env_, cfg.policy_path.c_str(), session_options);

  // Prepare input/output names (ORT >= 1.14: allocated strings must outlive the session run)
  io_name_holders_.push_back(session_->GetInputNameAllocated(0, allocator_));
  io_name_holders_.push_back(session_->GetOutputNameAllocated(0, allocator_));
  input_names = {io_name_holders_[0].get()};
  output_names = {io_name_holders_[1].get()};

  // Verify the model matches the compiled observation/action layout before touching the robot
  const auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto last_dim = [](const std::vector<int64_t>& shape) { return shape.empty() ? int64_t(-1) : shape.back(); };

  std::cout << "[ONNX] input '" << input_names[0] << "' dim=" << last_dim(in_shape)
            << ", output '" << output_names[0] << "' dim=" << last_dim(out_shape) << std::endl;

  if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
    throw std::runtime_error("ONNX model must have exactly 1 input and 1 output (no recurrent state supported)");
  if (last_dim(in_shape) != static_cast<int64_t>(NUM_OBS))
    throw std::runtime_error("ONNX input dim " + std::to_string(last_dim(in_shape)) + " != NUM_OBS " + std::to_string(NUM_OBS));
  if (last_dim(out_shape) != static_cast<int64_t>(NUM_ACTIONS))
    throw std::runtime_error("ONNX output dim " + std::to_string(last_dim(out_shape)) + " != NUM_ACTIONS " + std::to_string(NUM_ACTIONS));
}


/*****************************************************************************
** DDS Callback functions
*****************************************************************************/
void WholeBodyRL::LowStateHandler(const void *message)
{
  LowState_ low_state = *(const LowState_ *)message;
  if (low_state.crc() != g1_sim2real::Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) 
  {
    std::cout << "[ERROR] CRC Error" << std::endl;
    return;
  }

  // get motor state
  MotorState ms_tmp;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) 
  {
    ms_tmp.q.at(i) = low_state.motor_state()[i].q();
    ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    ms_tmp.tau.at(i) = low_state.motor_state()[i].tau_est();
  }
  motor_state_buffer_.SetData(ms_tmp);

  // get imu state
  ImuState imu_tmp;
  imu_tmp.omega = low_state.imu_state().gyroscope();
  imu_tmp.rpy = low_state.imu_state().rpy();
  imu_tmp.quat = low_state.imu_state().quaternion();
  imu_state_buffer_.SetData(imu_tmp);

  // update gamepad: from the xbox controller when selected, otherwise from the wireless remote.
  // The remote's Select stays live as an emergency stop either way.
  memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
  if (xbox_ && xbox_->connected())
  {
    auto xbox_data = xbox_->data();
    gamepad_.update(xbox_data);
    if (rx_.RF_RX.btn.components.select) should_exit_ = true;
  }
  else
  {
    // wireless remote; also the fallback while an xbox pad is selected but unplugged
    gamepad_.update(rx_.RF_RX);
  }

  // update mode machine
  if (mode_machine_ != low_state.mode_machine()) 
  {
    if (mode_machine_ == 0) std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
    mode_machine_ = low_state.mode_machine();
  }

  if (static_cast<int>(gamepad_.select.pressed) == 1)
    should_exit_ = true;

  // report robot status every second
  if (++counter_ % 500 == 0) 
  {
    counter_ = 0;

    // IMU
    auto &rpy = low_state.imu_state().rpy();
    // printf("IMU.pelvis.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);

    // RC
    // printf("gamepad_.A.pressed: %d\n", static_cast<int>(gamepad_.A.pressed));
    // printf("gamepad_.B.pressed: %d\n", static_cast<int>(gamepad_.B.pressed));
    // printf("gamepad_.X.pressed: %d\n", static_cast<int>(gamepad_.X.pressed));
    // printf("gamepad_.Y.pressed: %d\n", static_cast<int>(gamepad_.Y.pressed));

    // printf("gamepad_.go vertical: %f\n", static_cast<float>(gamepad_.ly));
    // printf("gamepad_.go horizontal: %f\n", static_cast<float>(gamepad_.lx));
    // printf("gamepad_.turn left: %f\n", static_cast<float>(gamepad_.rx));
    // printf("gamepad_.turn right: %f\n", static_cast<float>(gamepad_.ry));

    // Motor
    auto &ms = low_state.motor_state();
    // printf("All %d Motors:", G1_NUM_MOTOR);
    // printf("\nmode: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%u,", ms[i].mode());
    // printf("\npos: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%.2f,", ms[i].q());
    // printf("\nvel: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%.2f,", ms[i].dq());
    // printf("\ntau_est: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%.2f,", ms[i].tau_est());
    // printf("\ntemperature: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%d,%d;", ms[i].temperature()[0], ms[i].temperature()[1]);
    // printf("\nvol: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%.2f,", ms[i].vol());
    // printf("\nsensor: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%u,%u;", ms[i].sensor()[0], ms[i].sensor()[1]);
    // printf("\nmotorstate: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%u,", ms[i].motorstate());
    // printf("\nreserve: ");
    // for (int i = 0; i < G1_NUM_MOTOR; ++i) printf("%u,%u,%u,%u;", ms[i].reserve()[0], ms[i].reserve()[1], ms[i].reserve()[2], ms[i].reserve()[3]);
    // printf("\n");
  }
}

void WholeBodyRL::imuTorsoHandler(const void *message) 
{
  IMUState_ imu_torso = *(const IMUState_ *)message;
  auto &rpy = imu_torso.rpy();
  // if (counter_ % 500 == 0)
  //   printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
}


/*****************************************************************************
** DDS Command Writer function
*****************************************************************************/
void WholeBodyRL::LowCommandWriter()
{
  std::array<float, G1_NUM_MOTOR> torque_des;

  command_writer_freq_monitor.Tick();

  LowCmd_ dds_low_command;
  dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
  dds_low_command.mode_machine() = mode_machine_;

  const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

  if (mc && ms)
  {
    for (size_t i = 0; i < G1_NUM_MOTOR; i++)
    {
      dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
      dds_low_command.motor_cmd().at(i).tau()  = mc->tau_ff.at(i);
      dds_low_command.motor_cmd().at(i).q()    = mc->q_target.at(i);
      dds_low_command.motor_cmd().at(i).dq()   = mc->dq_target.at(i);
      dds_low_command.motor_cmd().at(i).kp()   = mc->kp.at(i);
      dds_low_command.motor_cmd().at(i).kd()   = mc->kd.at(i);
    }

    // clamp torque a few Nm below the kill threshold, so a saturating command never trips
    // CheckSafetyLimits by itself (see TORQUE_CLAMP_RATIO)
    for (size_t i = 0; i < NUM_ACTIONS; i++)
    {
      const float clamp_limit = torque_limit.at(i) * TORQUE_CLAMP_RATIO;

      torque_des.at(i) = mc->kp.at(i) * (mc->q_target.at(i) - ms->q.at(i)) + mc->kd.at(i) * (mc->dq_target.at(i) - ms->dq.at(i));

      if (abs(torque_des.at(i)) > clamp_limit)  // torque command
      {
        clamp_mask_.fetch_or(1u << i, std::memory_order_relaxed);
        torque_des.at(i) = std::clamp(torque_des.at(i), -clamp_limit, clamp_limit);
        dds_low_command.motor_cmd().at(i).tau() = torque_des.at(i);
        dds_low_command.motor_cmd().at(i).q()   = 0;
        dds_low_command.motor_cmd().at(i).dq()  = 0;
        dds_low_command.motor_cmd().at(i).kp()  = 0.00001;
        dds_low_command.motor_cmd().at(i).kd()  = 0.00001;
      }
    }

    dds_low_command.crc() = g1_sim2real::Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
    lowcmd_publisher_->Write(dds_low_command);
  }

  CheckSafetyLimits();

  if (should_exit_) std::exit(0);

  if (state_ == State::RL_POLICY_ACTIVE) cnt++;
}


/*****************************************************************************
** Control functions
*****************************************************************************/
void WholeBodyRL::Control() 
{
  // monitor control freq monitor
  control_freq_monitor.Tick();
  
  MotorCommand motor_command_tmp;
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

  for (int i = 0; i < G1_NUM_MOTOR; ++i) 
  {
    motor_command_tmp.q_target.at(i) = 0.0;
    motor_command_tmp.dq_target.at(i) = 0.0;
    motor_command_tmp.kp.at(i) = 0;
    motor_command_tmp.kd.at(i) = 8;
    motor_command_tmp.tau_ff.at(i) = 0.0;
  }

  if (ms)
  {
    time_abs += command_dt_;

    if (keyboard_) UpdateKeyboardInput();
    if (!keyboard_ || combine_) EchoJoystickCommand();

    switch (state_) 
    {
      case State::WAIT_FOR_INIT_COMMAND:
      {
        // [Stage 0]: set robot to wait for init command
        bool start_requested = (static_cast<int>(gamepad_.start.pressed) == 1);
        if (kb_start_) { kb_start_ = false; start_requested = true; }
        if (sim_auto_start_)
        {
          sim_state_time_ += command_dt_;
          if (sim_state_time_ >= sim_auto_start_delay_) start_requested = true;
        }

        if (start_requested)
        {
          sim_state_time_ = 0.0f;
          std::cout << "[INFO] Start button pressed!" << std::endl;
          state_ = State::MOVING_TO_DEFAULT;
          printf("G1 is in [MOVING_TO_DEFAULT] mode!\n");
        }

        for (int i = 0; i < G1_NUM_MOTOR; ++i) 
        {
          motor_command_tmp.q_target.at(i) = ms->q.at(i);
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = 0;
          motor_command_tmp.kd.at(i) = 8;
          motor_command_tmp.tau_ff.at(i) = 0.0;
        }

        CheckSafetyLimits();
        
        break;
      }

      case State::MOVING_TO_DEFAULT:
      {
        // [Stage 1]: set robot to default posture
        time_ += command_dt_;

        if (time_ < duration_) 
        {
          for (int i = 0; i < G1_NUM_MOTOR; ++i) 
          {
            double ratio = std::clamp(time_ / duration_, 0.f, 1.f);
            motor_command_tmp.q_target.at(i) = (1.0 - ratio) * ms->q.at(i) + ratio * cfg.default_pos[i];
            motor_command_tmp.kp.at(i) = cfg.init_kp[i];
            motor_command_tmp.kd.at(i) = cfg.init_kd[i];
          }
        }
        else if (time_ >= duration_)
        {
          std::cout << "[INFO] Reached default pos.\n";
          state_ = State::WAIT_FOR_POLICY_COMMAND;

          for (int i = 0; i < G1_NUM_MOTOR; ++i)
          {
            motor_command_tmp.q_target.at(i) = cfg.default_pos[i];
            motor_command_tmp.kp.at(i) = cfg.init_kp[i];
            motor_command_tmp.kd.at(i) = cfg.init_kd[i];
          }
        }

        CheckSafetyLimits();

        break;
      }

      case State::WAIT_FOR_POLICY_COMMAND:
      {
        // [Stage 2]: set robot to wait for RL command
        logging_active_ = true;

        bool policy_requested = (static_cast<int>(gamepad_.A.pressed) == 1);
        if (kb_policy_) { kb_policy_ = false; policy_requested = true; }
        if (sim_auto_start_)
        {
          sim_state_time_ += command_dt_;
          if (sim_state_time_ >= 2.0f) policy_requested = true;  // settle in the default pose first
        }

        if (policy_requested)
        {
          sim_state_time_ = 0.0f;
          time_ = 0.0;
          ResetPolicyState();
          std::cout << "[INFO] A button pressed!" << std::endl;
          state_ = State::RL_POLICY_ACTIVE;
        }

        for (int i = 0; i < G1_NUM_MOTOR; ++i)
        {
          motor_command_tmp.q_target.at(i) = cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.init_kp[i];
          motor_command_tmp.kd.at(i) = cfg.init_kd[i];
        }

        CheckSafetyLimits();

        break;
      }

      case State::RL_POLICY_ACTIVE:
      {
        // [Stage 3]: run robot with RL policy
        if (static_cast<int>(gamepad_.select.pressed) == 1)
        {
          std::cout << "[INFO] select button pressed!" << std::endl;
          state_ = State::DAMPING_STATE;
        }

        rl_action_ = RunInference();

        mode_pr_ = Mode::PR;

        // whole-body: policy output (policy joint order) -> SDK motor via joint_ids_map
        for (size_t i = 0; i < NUM_ACTIONS; ++i)
        {
          const int m = joint_ids_map[i];
          motor_command_tmp.q_target.at(m) = (rl_action_.at(i) * cfg.action_scale) + cfg.default_pos[m];
          motor_command_tmp.kp.at(m) = cfg.rl_kp[m];
          motor_command_tmp.kd.at(m) = cfg.rl_kd[m];
        }

        CheckSafetyLimits();

        break;
      }

      case State::DAMPING_STATE:
      {
        // [Stage 4]: finish robot control
        logging_active_ = false;

        for (int i = 0; i < G1_NUM_MOTOR; ++i)
        {
          motor_command_tmp.q_target.at(i) = 0;
          motor_command_tmp.dq_target.at(i) = 0;
          motor_command_tmp.kp.at(i) = 0;
          motor_command_tmp.kd.at(i) = 8;
          motor_command_tmp.tau_ff.at(i) = 0;
        }

        should_exit_ = true;

        break;
      }
    }
  }
  else
  {
    std::cerr << "[WARNING] MotorState not available!\n";
  }

  motor_command_buffer_.SetData(motor_command_tmp);
}

void WholeBodyRL::CheckSafetyLimits()
{
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

  if (ms)
  {
    for (int i = 0; i < G1_NUM_MOTOR; ++i)
    {
      if (ms->q.at(i) < joint_pos_min.at(i) - JOINT_POS_MARGIN // check joint position limit (min)
          || ms->q.at(i) > joint_pos_max.at(i) + JOINT_POS_MARGIN // check joint position limit (max)
          || abs(ms->tau.at(i)) > torque_limit.at(i)) // check joint torque limit
      {
        std::cout<< "\033[31m[ERROR] Motor state limitation Occur! \033[0m\n";
        std::cout<< "[INFO] State will be changed DAMPING_STATE \n";

        std::cout << " > joint_pos " << i << " : " << ms->q.at(i) << std::endl;
        std::cout << " > torque    " << i << " : " << ms->tau.at(i) << std::endl;

        state_ = State::DAMPING_STATE;
        should_exit_=true;

        break;
      }
    }

    for (int i = G1_NUM_LEG_MOTOR; i < G1_NUM_MOTOR; ++i) // for upper body
    {
      if (abs(ms->dq.at(i)) > joint_vel_limit) // check joint velocity limit
      {
        std::cout<< "\033[31m[ERROR] Motor state limitation Occur! \033[0m\n";
        std::cout<< "[INFO] State will be changed DAMPING_STATE \n";

        std::cout << " > joint_vel " << i << " : " << ms->dq.at(i) << std::endl;

        state_ = State::DAMPING_STATE;
        should_exit_=true;

        break;
      }
    }
  }
}


/*****************************************************************************
** RL functions
*****************************************************************************/
void WholeBodyRL::UpdateKeyboardInput()
{
  const std::string key = keyboard_->key();
  const bool pressed = !key.empty() && key != kb_last_key_;  // edge, so a held key does not run away
  kb_last_key_ = key;

  kb_idle_ = pressed ? 0.0f : kb_idle_ + command_dt_;

  if (pressed)
  {
    if      (key == "1") kb_start_ = true;
    else if (key == "2") kb_policy_ = true;
    else if (key == "x") { state_ = State::DAMPING_STATE; }
    else if (key == " ") kb_cmd_ = {};
    else if (key == "w") kb_cmd_[0] += KB_CMD_STEP;
    else if (key == "s") kb_cmd_[0] -= KB_CMD_STEP;
    else if (key == "a") kb_cmd_[1] += KB_CMD_STEP;
    else if (key == "d") kb_cmd_[1] -= KB_CMD_STEP;
    else if (key == "q") kb_cmd_[2] += KB_CMD_STEP;
    else if (key == "e") kb_cmd_[2] -= KB_CMD_STEP;

    for (int i = 0; i < 3; ++i) kb_cmd_[i] = std::clamp(kb_cmd_[i], cfg.cmd_min[i], cfg.cmd_max[i]);

    if (state_ == State::RL_POLICY_ACTIVE && key != "1" && key != "2")
      printf("[INPUT] cmd: vx %.2f  vy %.2f  wz %.2f\n", kb_cmd_[0], kb_cmd_[1], kb_cmd_[2]);
  }

  // Coast to a stop if the operator stops typing: a lost terminal focus should not leave the robot
  // running at speed. Set keyboard_idle_timeout to 0 in the config to switch this off.
  // Not in combine mode: there the keyboard command is a held cruise setting by design, and the
  // joystick in the operator's hand is the live stop.
  if (!combine_ && cfg.keyboard_idle_timeout > 0.0f && kb_idle_ > cfg.keyboard_idle_timeout)
  {
    bool moving = false;
    for (int i = 0; i < 3; ++i)
    {
      const float step = cfg.cmd_max[i] * command_dt_ / KB_DECAY_S;
      if (std::abs(kb_cmd_[i]) <= step) kb_cmd_[i] = 0.0f;
      else { kb_cmd_[i] -= std::copysign(step, kb_cmd_[i]); moving = true; }
    }
    if (moving && static_cast<int>(kb_idle_ / command_dt_) % 25 == 0)
      printf("[INPUT] idle %.1fs -- coasting to a stop\n", kb_idle_);
  }
}

void WholeBodyRL::EchoJoystickCommand()
{
  // Echo the stick command (wireless remote / xbox) so the operator can verify the joystick
  // works and see what the policy is being told: 1 Hz while deflected, one zero line on release.
  // In combine mode this is the merged command (keyboard base + stick).
  std::array<float, 3> cmd = {};
  const std::array<float, 3> stick = { static_cast<float>(gamepad_.ly),
                                       static_cast<float>(-gamepad_.lx),
                                       static_cast<float>(-gamepad_.rx) };
  bool moving = false;
  for (int i = 0; i < 3; ++i)
  {
    const float base = combine_ ? kb_cmd_[i] : 0.0f;
    cmd[i] = std::clamp(base + stick[i] * cfg.cmd_max[i], cfg.cmd_min[i], cfg.cmd_max[i]);
    moving |= std::abs(cmd[i]) > 0.02f;
  }

  ++js_echo_cnt_;
  if ((moving && js_echo_cnt_ % 50 == 0) || (!moving && js_echo_moving_))
    printf("[INPUT] cmd: vx %.2f  vy %.2f  wz %.2f\n", cmd[0], cmd[1], cmd[2]);
  js_echo_moving_ = moving;
}

void WholeBodyRL::ResetPolicyState()
{
  // mirrors env.reset() in deployment: gait clock restarts, last_action = 0, history refilled on first step
  cnt = 0;
  rl_action_.fill(0.0f);
  obs_history_initialized_ = false;
}

std::array<float, WholeBodyRL::NUM_SINGLE_OBS> WholeBodyRL::GetSingleObservation()
{
  // one frame in training term order:
  //   base_ang_vel(3) | projected_gravity(3) | velocity_commands(3) | joint_pos_rel(29) | joint_vel_rel(29) | last_action(29) | gait_phase(2)
  // caller guarantees motor/imu buffers are valid
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
  const std::shared_ptr<const ImuState> is = imu_state_buffer_.GetData();

  std::array<float, NUM_SINGLE_OBS> obs = {};
  size_t idx = 0;

  // base_ang_vel
  for (int i = 0; i < 3; ++i)
    obs[idx++] = is->omega[i] * cfg.ang_vel_scale;

  // projected_gravity
  const std::array<float, 3> gravity_orientation = GetGravityOrientation(is->quat);
  for (int i = 0; i < 3; ++i)
    obs[idx++] = gravity_orientation[i];

  // velocity_commands: full stick = cmd_max, then clamp to the (asymmetric) training range
  const std::array<float, 3> stick = { static_cast<float>(gamepad_.ly),
                                       static_cast<float>(-gamepad_.lx),
                                       static_cast<float>(-gamepad_.rx) };
  // SIM_CMD is already in m/s and rad/s, so it bypasses the stick scaling but keeps the clamp
  const float sim_ramp = std::clamp(cnt * control_dt_ / SIM_CMD_RAMP_S, 0.0f, 1.0f);
  std::array<float, 3> cmd = {};
  for (int i = 0; i < 3; ++i)
  {
    // combine: keyboard holds a base command (cruise), the stick adds on top of it
    const float stick_cmd = stick[i] * cfg.cmd_max[i];
    const float raw = sim_cmd_active_ ? sim_cmd_[i] * sim_ramp
                    : combine_         ? kb_cmd_[i] + stick_cmd
                    : keyboard_        ? kb_cmd_[i]
                                       : stick_cmd;
    cmd[i] = std::clamp(raw, cfg.cmd_min[i], cfg.cmd_max[i]);
    obs[idx++] = cmd[i] * cfg.cmd_scale[i];
  }
  last_cmd_ = cmd;

  // joint_pos_rel (policy order)
  for (size_t i = 0; i < NUM_ACTIONS; ++i)
  {
    const int m = joint_ids_map[i];
    obs[idx++] = (ms->q[m] - cfg.default_pos[m]) * cfg.dof_pos_scale;
  }

  // joint_vel_rel (policy order)
  for (size_t i = 0; i < NUM_ACTIONS; ++i)
    obs[idx++] = ms->dq[joint_ids_map[i]] * cfg.dof_vel_scale;

  // last_action (raw policy output)
  for (size_t i = 0; i < NUM_ACTIONS; ++i)
    obs[idx++] = rl_action_[i];

  // gait_phase: clock runs on time since activation, masked to (0,0) while standing
  const float phase = std::fmod(cnt * control_dt_, cfg.gait_period) / cfg.gait_period;
  const float cmd_norm = std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1] + cmd[2] * cmd[2]);
  if (cmd_norm < cfg.cmd_threshold)
  {
    obs[idx++] = 0.0f;
    obs[idx++] = 0.0f;
  }
  else
  {
    obs[idx++] = std::sin(2.0f * static_cast<float>(M_PI) * phase);
    obs[idx++] = std::cos(2.0f * static_cast<float>(M_PI) * phase);
  }
  last_phase_ = { obs[idx - 2], obs[idx - 1] };

  return obs;
}

std::vector<float> WholeBodyRL::GetObservation()
{
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
  const std::shared_ptr<const ImuState> is = imu_state_buffer_.GetData();

  if (!ms || !is)
  {
    std::cerr << "[ERROR] Failed to make Observation! Change to Damping mode!" << std::endl;
    state_ = State::DAMPING_STATE;
    should_exit_ = true;
    return std::vector<float>(NUM_OBS, 0.0f);
  }

  const std::array<float, NUM_SINGLE_OBS> frame = GetSingleObservation();

  // history buffer: oldest -> newest; on (re)activation fill every slot with the first frame (IsaacLab CircularBuffer reset)
  if (!obs_history_initialized_)
  {
    obs_history_.fill(frame);
    obs_history_initialized_ = true;
  }
  else
  {
    std::rotate(obs_history_.begin(), obs_history_.begin() + 1, obs_history_.end());
    obs_history_.back() = frame;
  }

  // flatten term-major, oldest -> newest inside each term (IsaacLab flatten_history_dim layout):
  //   [ang_vel x5][gravity x5][cmd x5][joint_pos x5][joint_vel x5][last_action x5][phase x5]
  static constexpr std::array<std::pair<size_t, size_t>, 7> terms = {{
    {0, 3},
    {3, 3},
    {6, 3},
    {9, NUM_ACTIONS},
    {9 + NUM_ACTIONS, NUM_ACTIONS},
    {9 + 2 * NUM_ACTIONS, NUM_ACTIONS},
    {9 + 3 * NUM_ACTIONS, 2}
  }};

  std::vector<float> obs;
  obs.reserve(NUM_OBS);
  for (const auto& [offset, len] : terms)
    for (size_t h = 0; h < OBS_HISTORY_LEN; ++h)
      obs.insert(obs.end(), obs_history_[h].begin() + offset, obs_history_[h].begin() + offset + len);

  return obs;
}

std::array<float, WholeBodyRL::NUM_ACTIONS> WholeBodyRL::RunInference()
{
  // Get observation data
  input_data = GetObservation();

  // create the input tensor with observation data
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  // create input tensors (input_tensor)
  input_tensors.clear();
  input_tensors.push_back(std::move(input_tensor));

  // run the inference
  std::vector<Ort::Value> output_tensors;

  try 
  {
    output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), 1, output_names.data(), 1);
  } 
  catch (const std::exception& e) 
  {
    std::cerr << "Inference failed: " << e.what() << " -> Change to Damping mode!" << std::endl;
    state_ = State::DAMPING_STATE;
    should_exit_ = true;
    return rl_action_;  // hold the last action for the remaining frame
  }

  // process the output tensor
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
  if (output_size < NUM_ACTIONS)
  {
    std::cerr << "Inference output size " << output_size << " < NUM_ACTIONS -> Change to Damping mode!" << std::endl;
    state_ = State::DAMPING_STATE;
    should_exit_ = true;
    return rl_action_;
  }

  // convert output data to vector
  std::vector<float> result(output_data, output_data + output_size);

  // convert result data to array
  std::array<float, NUM_ACTIONS> output_action = {};
  for (size_t i = 0; i < NUM_ACTIONS; ++i)
  {
    output_action[i] = result[i];
  }

  return output_action;
}


/*****************************************************************************
** Helper functions
*****************************************************************************/
std::array<float, 3> WholeBodyRL::GetGravityOrientation(const std::array<float, 4>& q)
{
  float qw = q[0];
  float qx = q[1];
  float qy = q[2];
  float qz = q[3];

  std::array<float, 3> gravity_orientation;

  gravity_orientation[0] = 2.0f * (-qz * qx + qw * qy);
  gravity_orientation[1] = -2.0f * (qz * qy + qw * qx);
  gravity_orientation[2] = 1.0f - 2.0f * (qw * qw + qz * qz);

  return gravity_orientation;
}

std::array<float, 3> WholeBodyRL::Quat2RPY(const std::array<float, 4>& q)
{
  float qw = q[0];
  float qx = q[1];
  float qy = q[2];
  float qz = q[3];

  std::array<float, 3> rpy;

  rpy[0] = 2.0f * (-qz * qx + qw * qy);
  rpy[1] = -2.0f * (qz * qy + qw * qx);
  rpy[2] = 1.0f - 2.0f * (qw * qw + qz * qz);

  float r0 = 2.0f * (qw * qx + qy * qz);
  float r1 = 1.0f - 2.0f * (qx * qx + qy * qy);
  rpy[0] = std::atan2(r0, r1);

  float r2 = 2.0f * (qw * qy - qz * qx);
  r2 = std::clamp(r2, -1.0f, 1.0f);
  rpy[1] = std::asin(r2);

  float r3 = 2.0f * (qw * qz + qx * qy);
  float r4 = 1.0f - 2.0f * (qy * qy + qz * qz);
  rpy[2] = std::atan2(r3, r4);

  return rpy;
}


/*****************************************************************************
** main function
*****************************************************************************/
int main()
{
  WholeBodyRL wholebodyrl("../configs/g1_sim2real.yaml");

  while (true)
  {
    if (wholebodyrl.should_exit_) break;
    std::this_thread::sleep_for(std::chrono::microseconds(2000));  // 2ms = 2000μs
  }

  return 0;
}
