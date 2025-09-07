/******************************************************************************************
* HumARConoid-Sim2Real
*
* Sim2Real Transfer to Optimize Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid-Sim2Real.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************/

/* Authors: Sol Choi (Jennifer) */

#include "arc_g1/wholebody_arc.hpp"

const std::array<float, WholeBodyRL::G1_NUM_MOTOR> WholeBodyRL::Kp
{
  60, 60, 60, 100, 40, 40,      // left leg
  60, 60, 60, 100, 40, 40,      // right leg
  60, 40, 40,                   // waist
  40, 40, 40, 40,  40, 40, 40,  // left arm
  40, 40, 40, 40,  40, 40, 40   // right arm
};

const std::array<float, WholeBodyRL::G1_NUM_MOTOR> WholeBodyRL::Kd
{
  1, 1, 1, 2, 1, 1,     // left leg
  1, 1, 1, 2, 1, 1,     // right leg
  1, 1, 1,              // waist
  1, 1, 1, 1, 1, 1, 1,  // left arm
  1, 1, 1, 1, 1, 1, 1   // right arm
};

WholeBodyRL::WholeBodyRL(std::string networkInterface, const std::string& model_path)
: time_(0.0),
  control_dt_(0.002), // 500Hz
  command_dt_(0.02), // 50Hz
  duration_(5.0),
  counter_(0),
  mode_pr_(Mode::PR),
  mode_machine_(0),
  input_tensor_values_(num_obs),
  time_abs(0.0)
{
  // initialize unitree channel
  ChannelFactory::Instance()->Init(0, networkInterface);

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

  LoadYamlConfig(model_path);
  LoadOnnxModel();

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
  lowstate_subscriber_->InitChannel(std::bind(&WholeBodyRL::LowStateHandler, this, std::placeholders::_1), 1); // 1kHz

  imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
  imutorso_subscriber_->InitChannel(std::bind(&WholeBodyRL::imuTorsoHandler, this, std::placeholders::_1), 1); // 1kHz

  estimate_state_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(GO_ODOM_TOPIC));
  estimate_state_subscriber->InitChannel(std::bind(&WholeBodyRL::HighFreOdomMessageHandler, this, std::placeholders::_1), 1); // 500Hz
}

void WholeBodyRL::InitThread()
{
  // create threads
  command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000, &WholeBodyRL::LowCommandWriter, this); // 500Hz
  control_thread_ptr_ = CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, 20000, &WholeBodyRL::Control, this); // 50Hz

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

    if (49.99 <= control_hz && control_hz <= 50.01 && !safe_freq_ &&
        499.99 <= command_writer_hz && command_writer_hz <= 500.01)
    {
      safe_freq_ = true;
      std::cout << "\033[32m[Freq Monitor]\033[0m Control freq stable!" 
        << " (Control loop: " << control_hz << " Hz, " << "Command Writer loop: " << command_writer_hz << " Hz)"<< std::endl;
    }
    else if (49.99 >= control_hz && control_hz >= 50.01 &&
             499.99 >= command_writer_hz && command_writer_hz >= 500.01)
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
  std::ofstream log_file("../utility/sources/log_arc_rough5.csv");
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
      for (int i = 0; i < G1_NUM_MOTOR; ++i) log_file << "," << cfg.rl_kp.at(i) * (mc->q_target.at(i) - ms->q.at(i)) + cfg.rl_kd.at(i) * (0 - ms->dq.at(i));
      log_file << "\n";
      log_file.flush();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  log_file.close();
  std::cout << "[Logger] Stopped logging." << std::endl;
}

void WholeBodyRL::LoadYamlConfig(const std::string& config_yaml_path)
{
  YAML::Node config = YAML::LoadFile(config_yaml_path);

  const std::string var_name = "{ARC_G1_POLICY_DIR}";
  std::string replacement = "../policy";

  std::string model_path = config["onnx_path"].as<std::string>();

  size_t pos = model_path.find(var_name);
  if (pos != std::string::npos) model_path.replace(pos, var_name.length(), replacement);

  // onnx path
  cfg.model_path = model_path;

  // default pos
  std::vector<float> default_angles = config["default_angles"].as<std::vector<float>>();
  std::vector<float> arm_waist_target = config["arm_waist_target"].as<std::vector<float>>();
  if (default_angles.size() + arm_waist_target.size() != G1_NUM_MOTOR) throw std::runtime_error("YAML default_angles + arm_waist_target size mismatch");
  std::copy(default_angles.begin(), default_angles.end(), cfg.default_pos.begin());
  std::copy(arm_waist_target.begin(), arm_waist_target.end(), cfg.default_pos.begin() + default_angles.size());

  // init kp
  std::vector<float> init_leg_kps = config["init_leg_kps"].as<std::vector<float>>();
  std::vector<float> init_arm_waist_kps = config["init_arm_waist_kps"].as<std::vector<float>>();
  if (init_leg_kps.size() + init_arm_waist_kps.size() != G1_NUM_MOTOR) throw std::runtime_error("YAML init_leg_kps + init_arm_waist_kps size mismatch");
  std::copy(init_leg_kps.begin(), init_leg_kps.end(), cfg.init_kp.begin());
  std::copy(init_arm_waist_kps.begin(), init_arm_waist_kps.end(), cfg.init_kp.begin() + init_leg_kps.size());

  // init kd
  std::vector<float> init_leg_kds = config["init_leg_kds"].as<std::vector<float>>();
  std::vector<float> init_arm_waist_kds = config["init_arm_waist_kds"].as<std::vector<float>>();
  if (init_leg_kds.size() + init_arm_waist_kds.size() != G1_NUM_MOTOR) throw std::runtime_error("YAML init_leg_kds + init_arm_waist_kds size mismatch");
  std::copy(init_leg_kds.begin(), init_leg_kds.end(), cfg.init_kd.begin());
  std::copy(init_arm_waist_kds.begin(), init_arm_waist_kds.end(), cfg.init_kd.begin() + init_leg_kds.size());

  // RL kp
  std::vector<float> kps = config["kps"].as<std::vector<float>>();
  std::vector<float> arm_kps = config["arm_waist_kps"].as<std::vector<float>>();
  if (kps.size() + arm_kps.size() != G1_NUM_MOTOR) throw std::runtime_error("YAML kps + arm_waist_kps size mismatch");
  std::copy(kps.begin(), kps.end(), cfg.rl_kp.begin());
  std::copy(arm_kps.begin(), arm_kps.end(), cfg.rl_kp.begin() + kps.size());

  // RL kd
  std::vector<float> kds = config["kds"].as<std::vector<float>>();
  std::vector<float> arm_kds = config["arm_waist_kds"].as<std::vector<float>>();
  if (kds.size() + arm_kds.size() != G1_NUM_MOTOR) throw std::runtime_error("YAML kds + arm_waist_kds size mismatch");
  std::copy(kds.begin(), kds.end(), cfg.rl_kd.begin());
  std::copy(arm_kds.begin(), arm_kds.end(), cfg.rl_kd.begin() + kds.size());

  cfg.ang_vel_scale = config["ang_vel_scale"].as<float>();
  cfg.dof_pos_scale = config["dof_pos_scale"].as<float>();
  cfg.dof_vel_scale = config["dof_vel_scale"].as<float>();
  cfg.action_scale = config["action_scale"].as<float>();
  std::vector<float> cmd_scale = config["cmd_scale"].as<std::vector<float>>();
  std::copy(cmd_scale.begin(), cmd_scale.end(), cfg.cmd_scale.begin());
  std::vector<float> max_cmd = config["max_cmd"].as<std::vector<float>>();
  std::copy(max_cmd.begin(), max_cmd.end(), cfg.max_cmd.begin());
  cfg.num_actions = config["num_actions"].as<size_t>();
  cfg.num_obs = config["num_obs"].as<size_t>();

  PrintYamlConfig();
}

void WholeBodyRL::PrintYamlConfig()
{
  std::cout << "\n============================== [RL Config Loaded] ==============================\n";
  std::cout << "  Model path       : " << cfg.model_path << "\n";

  // std::cout << "  Init Kp size     : " << cfg.init_kp.size() << "\n";
  // std::cout << "  Init Kd size     : " << cfg.init_kd.size() << "\n";
  // std::cout << "  RL Kp size       : " << cfg.rl_kp.size() << "\n";
  // std::cout << "  RL Kd size       : " << cfg.rl_kd.size() << "\n";

  std::cout << "  Scale factors:\n";
  std::cout << "    ang_vel_scale  : " << cfg.ang_vel_scale << "\n";
  std::cout << "    dof_pos_scale  : " << cfg.dof_pos_scale << "\n";
  std::cout << "    dof_vel_scale  : " << cfg.dof_vel_scale << "\n";
  std::cout << "    action_scale   : " << cfg.action_scale << "\n";
  std::cout << "    cmd_scale      : [" << cfg.cmd_scale[0] << ", " << cfg.cmd_scale[1] << ", " << cfg.cmd_scale[2] << "]\n";
  std::cout << "    max_cmd        : [" << cfg.max_cmd[0] << ", " << cfg.max_cmd[1] << ", " << cfg.max_cmd[2] << "]\n";

  std::cout << "  Model dimensions:\n";
  std::cout << "    num_actions    : " << cfg.num_actions << "\n";
  std::cout << "    num_obs        : " << cfg.num_obs << "\n";
  std::cout << "================================================================================\n" << std::endl;
}

void WholeBodyRL::LoadOnnxModel()
{
  // initialize onnx runtime env
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_cpu_RL");

  // initialize input data
  input_data = std::vector<float>(47, 0.0f);  // 47
  input_shape = {1, 47};  // 47

  // h_in_data = std::vector<float>(64, 0.0f); 
  // c_in_data = std::vector<float>(64, 0.0f); 
  // hidden_shape = {1, 1, 64};

  // session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // load onnx model
  session_ = std::make_unique<Ort::Session>(env_, cfg.model_path.c_str(), session_options);

  // Prepare input/output names
  const char* input_name = session_->GetInputName(0, allocator_);
  // const char* h_in_name = session_->GetInputName(1, allocator_);
  // const char* c_in_name = session_->GetInputName(2, allocator_);
  const char* output_name = session_->GetOutputName(0, allocator_);
  // const char* h_out_name = session_->GetOutputName(1, allocator_);
  // const char* c_out_name = session_->GetOutputName(2, allocator_);

  // input/output names
  input_names = {input_name}; // , h_in_name, c_in_name};
  output_names = {output_name}; // , h_out_name, c_out_name};
}

void WholeBodyRL::imuTorsoHandler(const void *message) 
{
  IMUState_ imu_torso = *(const IMUState_ *)message;
  auto &rpy = imu_torso.rpy();
  // if (counter_ % 500 == 0)
  //   printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
}

void WholeBodyRL::LowStateHandler(const void *message)
{
  LowState_ low_state = *(const LowState_ *)message;
  if (low_state.crc() != arc_g1::Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) 
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

  // update gamepad
  memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
  gamepad_.update(rx_.RF_RX);

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

void WholeBodyRL::HighFreOdomMessageHandler(const void* message)
{
  unitree_go::msg::dds_::SportModeState_ odom_state = *(const unitree_go::msg::dds_::SportModeState_*)message;
  
  // if (odom_state.crc() != arc_g1::Crc32Core((uint32_t *)&odom_state, (sizeof(unitree_go::msg::dds_::SportModeState_) >> 2) - 1)) 
  // {
  //   std::cout << "[ERROR] CRC Error" << std::endl;
  //   return;
  // }

  OdometerState om_tmp;

  for (int i = 0; i < 3; ++i) 
  {
    om_tmp.position.at(i) = odom_state.position()[i];
    om_tmp.velocity.at(i) = odom_state.velocity()[i];
    om_tmp.euler.at(i) = odom_state.imu_state().rpy()[i];

    om_tmp.quaternion.at(i) = odom_state.imu_state().quaternion()[i];
  }

  om_tmp.quaternion.at(3) = odom_state.imu_state().quaternion()[3];
  om_tmp.yaw_speed = odom_state.yaw_speed();

  odom_state_buffer_.SetData(om_tmp);

  // if (high_odom_cnt_ % 50 == 0)
  // {
  //   std::cout << "High position info: " << std::endl;
  //   std::cout << "High x: " << estimator_state.position()[0] << std::endl;
  //   std::cout << "High y: " << estimator_state.position()[1] << std::endl;
  //   std::cout << "High z: " << estimator_state.position()[2] << std::endl;

  //   std::cout << "velocity info: " << std::endl;
  //   std::cout << "x: " << estimator_state.velocity()[0] << std::endl;
  //   std::cout << "y: " << estimator_state.velocity()[1] << std::endl;
  //   std::cout << "z: " << estimator_state.velocity()[2] << std::endl;
  // }

  // high_odom_cnt_++;
  
  // std::cout << "euler angle info: " << std::endl;
  // std::cout << "x: " << estimator_state.imu_state().rpy()[0] << std::endl;
  // std::cout << "y: " << estimator_state.imu_state().rpy()[1] << std::endl;
  // std::cout << "z: " << estimator_state.imu_state().rpy()[2] << std::endl;

  // std::cout << "yaw speed info: " << std::endl;
  // std::cout << estimator_state.yaw_speed() << std::endl;

  // std::cout << "Quaternion info: " << std::endl;
  // std::cout << "w: " << estimator_state.imu_state().quaternion()[0] << std::endl;
  // std::cout << "x: " << estimator_state.imu_state().quaternion()[1] << std::endl;
  // std::cout << "y: " << estimator_state.imu_state().quaternion()[2] << std::endl;
  // std::cout << "z: " << estimator_state.imu_state().quaternion()[3] << std::endl;
}

void WholeBodyRL::LowCommandWriter()
{
  std::array<float, G1_NUM_MOTOR> torque_des;

  command_writer_freq_monitor.Tick();

  LowCmd_ dds_low_command;
  dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
  dds_low_command.mode_machine() = mode_machine_;

  const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

  // for upper swing motion
  const std::shared_ptr<const ImuState> is = imu_state_buffer_.GetData();

  if (mc && ms)
  {
    for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
      dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
      dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
      dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
      dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
      dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
      dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);

      // // clamp torque limit
      // torque_des.at(i) = cfg.rl_kp.at(i) * (mc->q_target.at(i) - ms->q.at(i)) + cfg.rl_kd.at(i) * (0 - ms->dq.at(i))

      // if (abs(torque_des.at(i)) > torque_limit.at(i))  // torque command
      // {
      //   // std::cout << "\033[31m \033[0m\n";
      //   torque_des.at(i) = std::clamp(torque_des.at(i), -torque_limit.at(i), torque_limit.at(i));
      //   dds_low_command.motor_cmd().at(i).tau() = torque_des.at(i);
      //   dds_low_command.motor_cmd().at(i).q() = 0;
      //   dds_low_command.motor_cmd().at(i).dq() = 0;
      //   dds_low_command.motor_cmd().at(i).kp() = 0;
      //   dds_low_command.motor_cmd().at(i).kd() = 0;
      // }
    }

    // if (is && state_ == State::RL_POLICY_ACTIVE) // 500Hz
    // {
    //   std::array<float, 3> waistrpy = Quat2RPY(is->quat);
    //   dds_low_command.motor_cmd().at(13).q() = -waistrpy.at(0);
    //   dds_low_command.motor_cmd().at(14).q() = -waistrpy.at(1);

    //   dds_low_command.motor_cmd().at(13).kp() = 1000;
    //   dds_low_command.motor_cmd().at(13).kd() = 20;

    //   dds_low_command.motor_cmd().at(14).kp() = 1000;
    //   dds_low_command.motor_cmd().at(14).kd() = 20;
    // }

    // clamp torque limit
    for (size_t i = 0; i < 12; i++)
    {
      torque_des.at(i) = cfg.rl_kp.at(i) * (mc->q_target.at(i) - ms->q.at(i)) + cfg.rl_kd.at(i) * (0 - ms->dq.at(i));

      if (abs(torque_des.at(i)) > torque_limit.at(i))  // torque command
      {
        // std::cout << "\033[31m \033[0m\n";
        torque_des.at(i) = std::clamp(torque_des.at(i), -torque_limit.at(i), torque_limit.at(i));
        dds_low_command.motor_cmd().at(i).tau() = torque_des.at(i);
        dds_low_command.motor_cmd().at(i).q() = 0;
        dds_low_command.motor_cmd().at(i).dq() = 0;
        dds_low_command.motor_cmd().at(i).kp() = 0;
        dds_low_command.motor_cmd().at(i).kd() = 0;
      }
    }

    dds_low_command.crc() = arc_g1::Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
    lowcmd_publisher_->Write(dds_low_command);
  }

  CheckExit();

  if (should_exit_) std::exit(0);

  if (state_ == State::RL_POLICY_ACTIVE) cnt++;
}

void WholeBodyRL::CheckExit()
{
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
  // const std::shared_ptr<const ImuState> is = imu_state_buffer_.GetData();

  if (ms)
  {
    for (int i = 0; i < G1_NUM_MOTOR; ++i) // for wholebody
    {
      if (ms->q.at(i) < joint_pos_min.at(i) || ms->q.at(i) > joint_pos_max.at(i)
          || abs(ms->tau.at(i)) > torque_limit.at(i)
      )
      {
        cout<< "\033[31m[ERROR] Motor state limitation Occur! \033[0m\n";
        cout<< "[INFO] State will be changed DAMPING_STATE \n";

        cout << " > joint_pos " << i << " : " << ms->q.at(i) << endl;
        cout << " > torque : " << i << " : " << ms->tau.at(i) << endl;

        state_ = State::DAMPING_STATE;
        should_exit_=true;

        break;
      }
    }
    for (int i = 12; i < G1_NUM_MOTOR; ++i) // for upper body
    {
      if (abs(ms->dq.at(i)) > joint_vel_limit)
      {
        cout<< "\033[31m[ERROR] Motor state limitation Occur! \033[0m\n";
        cout<< "[INFO] State will be changed DAMPING_STATE \n";

        cout << " > joint_vel " << i << " : " << ms->dq.at(i) << endl;

        state_ = State::DAMPING_STATE;
        should_exit_=true;

        break;
      }
    }
  }
}

void WholeBodyRL::Global2NewWorld()
{
  const std::shared_ptr<const OdometerState> os = odom_state_buffer_.GetData();

  if (os)
  {
    // std::cout << os->position[0] << ", " << os->position[1] << ", " << os->position[2] << std::endl;
    // Mat4 T = Mat4::identity();

    std::array<float, 4> quat = normalize_quat(os->quaternion);

    float gamma = atan2(2.f * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2.f * (quat[2] * quat[2] + quat[3] * quat[3]));

    float cos_gamma = cos(gamma);
    float sin_gamma = sin(gamma);

    // std::array<float, 9> transf_rot = {
    //   cos_gamma, -sin_gamma,   0.0f,
    //   sin_gamma,  cos_gamma,   0.0f,
    //        0.0f,       0.0f,   1.0f
    // };

    // std::array<float, 3> transf_trans = {os->position[0], os->position[1], 0.f};

    transf_global2nworld = {
       cos_gamma, sin_gamma, 0.f, -cos_gamma * os->position[0] - sin_gamma * os->position[1],
      -sin_gamma, cos_gamma, 0.f,  sin_gamma * os->position[0] - cos_gamma * os->position[1],
             0.f,       0.f, 1.f,                                                        0.f,
             0.f,       0.f, 0.f,                                                        1.f
    };

    // std::cout << "B2N" << std::endl;
  }
}

std::array<float, 16> WholeBodyRL::NewWorld2Base()
{
  const std::shared_ptr<const OdometerState> os = odom_state_buffer_.GetData();

  if (os)
  {
    std::array<float, 3> pos = {os->position[0], os->position[1], os->position[2]};
    std::array<float, 4> quat = normalize_quat(os->quaternion);

    std::array<float, 16> T_base = {
      1.f - 2.f * (quat[2] * quat[2] + quat[3] * quat[3]), 2.f * (quat[1] * quat[2] - quat[0] * quat[3]), 2.f * (quat[1] * quat[3] + quat[0] * quat[2]), pos[0],
      2.f * (quat[1] * quat[2] + quat[0] * quat[3]), 1.f - 2.f * (quat[1] * quat[1] + quat[3] * quat[3]), 2.f * (quat[2] * quat[3] - quat[0] * quat[1]), pos[1],
      2.f * (quat[1] * quat[3] - quat[0] * quat[2]), 2.f * (quat[2] * quat[3] + quat[0] * quat[1]), 1.f - 2.f * (quat[1] * quat[1] + quat[2] * quat[2]), pos[2],
      0.f, 0.f, 0.f, 1.f
    };

    std::array<float, 16> T_res = mat4_mul(transf_global2nworld, T_base);

    // std::cout << T_res[0] << "  " << T_res[1] << "  " << T_res[2] << "  " << T_res[3] << std::endl;
    // std::cout << T_res[4] << "  " << T_res[5] << "  " << T_res[6] << "  " << T_res[7] << std::endl;
    // std::cout << T_res[8] << "  " << T_res[9] << "  " << T_res[10] << "  " << T_res[11] << std::endl;
    // std::cout << T_res[12] << "  " << T_res[13] << "  " << T_res[14] << "  " << T_res[15] << std::endl;

    // std::cout << "                                                                                                     " << std::endl;

    // std::cout << "x_o: " << pos[0] << "y_o: " << pos[1] << "z_o: " << pos[2] << endl;
    // std::cout << "x: " << T_res[3] << "y: " << T_res[7] << "z: " << T_res[11] << endl;

    return T_res;
  }
}

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

std::vector<float> WholeBodyRL::GetMockObservation()
{
  std::vector<float> mock_obs(47, 0.0f);  // 47
  mock_obs[3] = 0.0f;  // gravity x
  mock_obs[4] = 0.0f;  // gravity y
  mock_obs[5] = 1.0f;  // gravity z

  mock_obs[9 + LeftKnee] = 0.2f;  // joint pos example
  return mock_obs;
}

std::vector<float> WholeBodyRL::GetObservation()
{
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
  const std::shared_ptr<const ImuState> is = imu_state_buffer_.GetData();
  const std::shared_ptr<const OdometerState> os = odom_state_buffer_.GetData();

  if (os)
  {
    transf_nworld2base = NewWorld2Base();
    yaw_rel = atan2(transf_nworld2base[4], transf_nworld2base[0]);
    p_y = transf_nworld2base[4] * os->position[0] + transf_nworld2base[5] * os->position[1] 
        + transf_nworld2base[6] * os->position[2] + transf_nworld2base[7];    
  }
  
  if (ms && is)
  {
    // cnt++;

    std::vector<float> obs(47);  // 47
    std::array<float, 3> gravity_orientation = GetGravityOrientation(is->quat);

    // ang_vel
    for (int i = 0; i < 3; ++i)
    {
      obs[i] = is->omega[i] * cfg.ang_vel_scale;
    }
    
    // projected gravity
    obs[3] = gravity_orientation[0];
    obs[4] = gravity_orientation[1];
    obs[5] = gravity_orientation[2];

    // command
    obs[6] = static_cast<float>(gamepad_.ly) * cfg.cmd_scale[0] * cfg.max_cmd[0];
    obs[7] = 0.f; // static_cast<float>(gamepad_.lx) * -1 * cfg.cmd_scale[1] * cfg.max_cmd[1]; // std::clamp(-p_y, -0.2f, 0.2f); 
    obs[8] = static_cast<float>(gamepad_.rx) * -1 * cfg.cmd_scale[2] * cfg.max_cmd[2]; // -1.2 * yaw_rel;

    if (state_ == State::RL_POLICY_WAVE_HAND)
    {
      obs[6] = 0.f;
      obs[7] = 0.f;
    }

    // obs[8] = -0.5 * estimator_state.imu_state().rpy()[2]; // static_cast<float>(gamepad_.rx) * -1 * cfg.cmd_scale[2] * cfg.max_cmd[2];

    // std::cout << "z: " << estimator_state.imu_state().rpy()[2] << std::endl;
    // std::cout << "cmd yaw vel: " << obs[8] << std::endl;

    // joint pos
    for (size_t i = 0; i < cfg.num_actions; ++i) // 12
    {
      int m2i_idx = mujoco2isaaclab[i];
      obs[start_idx + i] = (ms->q[m2i_idx] - cfg.default_pos[m2i_idx]) * cfg.dof_pos_scale;
    }

    // joint vel
    start_idx += cfg.num_actions;
    for (size_t i = 0; i < cfg.num_actions; ++i) // 12
    {
      int m2i_idx = mujoco2isaaclab[i];
      obs[start_idx + i] = ms->dq[m2i_idx] * cfg.dof_vel_scale;
    }

    // action
    std::copy(rl_action_.begin(), rl_action_.end(), obs.begin() + start_idx + cfg.num_actions); // 12

    float cmd_speed = std::sqrt(obs[6] * obs[6] + obs[7] * obs[7]);
    float stride_length = stride_a + stride_b * cmd_speed;
    float period = stride_length / (cmd_speed + eps);
    float phase = std::fmod(cnt * control_dt_, period) / period;

    if (cmd_speed > 0.1)
    {
      // sin phase
      obs[45] = std::sin(2.0 * M_PI * phase);

      // cos phase
      obs[46] = std::cos(2.0 * M_PI * phase);
    }
    else
    {
      obs[45] = 0.0f;
      obs[46] = 0.0f;
    }

    start_idx = 9;

    // upper: arm swing motion
    arm_swing_motion = arm_swing_action(phase, cmd_speed, 0.5);
    // arm_swing_motion = arm_swing_action(phase, cmd_speed, 0.3);

    command_speed = cmd_speed;

    return obs;
  }

  if (!ms || !is) 
  {
    // std::cerr << "[WARNING] Using mock observation." << std::endl;
    std::vector<float> mock_obs(47, 0.0f);  // 47
    mock_obs[3] = 0.0f;  // gravity x
    mock_obs[4] = 0.0f;  // gravity y
    mock_obs[5] = 1.0f;  // gravity z
    return mock_obs;
  }

  return std::vector<float>(47, 0.0f);  // 47
}

std::array<float, 12> WholeBodyRL::RunInference()
{
  // Get observation data
  input_data = GetObservation();

  // create the input tensor with observation data
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

  // create tensors for h_in and c_in
  // Ort::Value h_in_tensor = Ort::Value::CreateTensor<float>(memory_info, h_in_data.data(), h_in_data.size(), hidden_shape.data(), hidden_shape.size());
  // Ort::Value c_in_tensor = Ort::Value::CreateTensor<float>(memory_info, c_in_data.data(), c_in_data.size(), hidden_shape.data(), hidden_shape.size());

  // create input tensors (input_tensor, h_in_tensor, c_in_tensor)
  input_tensors.clear();
  input_tensors.push_back(std::move(input_tensor));
  // input_tensors.push_back(std::move(h_in_tensor));
  // input_tensors.push_back(std::move(c_in_tensor));

  // run the inference
  std::vector<Ort::Value> output_tensors;
  try 
  {
    output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), 1, output_names.data(), 1);
  } 
  catch (const std::exception& e) 
  {
    std::cerr << "Inference failed: " << e.what() << std::endl;
  }

  // process the output tensor
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

  // float* h_out_data = output_tensors[1].GetTensorMutableData<float>();
  // size_t h_out_size = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();

  // float* c_out_data = output_tensors[2].GetTensorMutableData<float>();
  // size_t c_out_size = output_tensors[2].GetTensorTypeAndShapeInfo().GetElementCount();

  // update the h_in and c_in states
  // std::copy(h_out_data, h_out_data + h_out_size, h_in_data.begin());
  // std::copy(c_out_data, c_out_data + c_out_size, c_in_data.begin());

  // convert output data to vector
  std::vector<float> result(output_data, output_data + output_size);

  // convert result data to array
  std::array<float, 12> output_action = {};
  for (size_t i = 0; i < 12; ++i) {
      output_action[i] = result[i];
  }

  return output_action;
}

std::array<float, 2> WholeBodyRL::arm_swing_action(float leg_phase, float speed, float amplitude)
{
  // Anti-phase for arms: shift by 0.5
  float arm_phase = std::fmod(leg_phase + 0.5, 1.0);

  // Detect double stance phase and apply damping
  float damping = 1.0f;

  if (0.0 <= leg_phase && leg_phase < 0.1)
  {
    damping = 0.5 * (1.0 - std::cos(M_PI * leg_phase / 0.1)); // 0 → 1
  } 
  else if (0.5 <= leg_phase && leg_phase < 0.6)
  {
    damping = 0.5 * (1.0 + std::cos(M_PI * (leg_phase - 0.5) / 0.1)); // 1 → 0
  }

  float speed_scale = std::clamp(speed / 0.2, 0.0, 1.0);
  float arm_angle = amplitude * damping * speed_scale * std::sin(2 * M_PI * arm_phase);

  return {arm_angle, -arm_angle};
}

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

    // rl_action_ = RunInference();
    // cout << "rl_action_ calced!" << endl;
    // cout << "cnt" << cnt << endl;

    switch (state_) 
    {
      case State::WAIT_FOR_INIT_COMMAND:
      {
        // [Stage 0]: set robot to wait for init command
        if (static_cast<int>(gamepad_.start.pressed) == 1)
        {
          std::cout << "[INFO] Start button pressed!" << std::endl;
          state_ = State::MOVING_TO_DEFAULT;
          printf("G1 is in [MOVING_TO_DEFAULT] mode!\n");
        }

        for (int i = 0; i < G1_NUM_MOTOR; ++i) 
        {
          motor_command_tmp.q_target.at(i) = ms->q.at(i);
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = Kp[i];
          motor_command_tmp.kd.at(i) = Kd[i];
          motor_command_tmp.tau_ff.at(i) = 0.0;
        }


        // // HERE: make below as function!
        // Global2NewWorld();

        // const std::shared_ptr<const OdometerState> os = odom_state_buffer_.GetData();
        // if (os)
        // {
        //   std::array<float, 3> pos = {os->position[0], os->position[1], os->position[2]};
        //   std::array<float, 4> quat = normalize_quat(os->quaternion);

        //   std::array<float, 16> T_base = {
        //     1.f - 2.f * (quat[2] * quat[2] + quat[3] * quat[3]), 2.f * (quat[1] * quat[2] - quat[0] * quat[3]), 2.f * (quat[1] * quat[3] + quat[0] * quat[2]), pos[0],
        //     2.f * (quat[1] * quat[2] + quat[0] * quat[3]), 1.f - 2.f * (quat[1] * quat[1] + quat[3] * quat[3]), 2.f * (quat[2] * quat[3] - quat[0] * quat[1]), pos[1],
        //     2.f * (quat[1] * quat[3] - quat[0] * quat[2]), 2.f * (quat[2] * quat[3] + quat[0] * quat[1]), 1.f - 2.f * (quat[1] * quat[1] + quat[2] * quat[2]), pos[2],
        //     0.f, 0.f, 0.f, 1.f
        //   };

        //   std::array<float, 16> T_res = mat4_mul(transf_global2nworld, T_base);
        //   std::cout << T_res[0] << "  " << T_res[1] << "  " << T_res[2] << "  " << T_res[3] << std::endl;
        //   std::cout << T_res[4] << "  " << T_res[5] << "  " << T_res[6] << "  " << T_res[7] << std::endl;
        //   std::cout << T_res[8] << "  " << T_res[9] << "  " << T_res[10] << "  " << T_res[11] << std::endl;
        //   std::cout << T_res[12] << "  " << T_res[13] << "  " << T_res[14] << "  " << T_res[15] << std::endl;

        //   std::cout << "                                                                                                     " << std::endl;
        // }

        CheckExit();
        
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

        CheckExit();

        break;
      }

      case State::WAIT_FOR_POLICY_COMMAND:
      {
        // [Stage 2]: set robot to wait for RL command
        logging_active_ = true;

        if (static_cast<int>(gamepad_.A.pressed) == 1)
        {
          time_ = 0.0;
          std::cout << "[INFO] A button pressed!" << std::endl;
          Global2NewWorld();
          state_ = State::RL_POLICY_ACTIVE;
        }

        if (static_cast<int>(gamepad_.Y.pressed) == 1)
        {
          time_ = 0.0;
          duration_ = 2.0;
          std::cout << "[INFO] Y button pressed!" << std::endl;
          Global2NewWorld();
          state_ = State::RL_POLICY_WAVE_HAND;
        }

        for (int i = 0; i < G1_NUM_MOTOR; ++i)
        {
          motor_command_tmp.q_target.at(i) = cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.init_kp[i];
          motor_command_tmp.kd.at(i) = cfg.init_kd[i];
        }

        CheckExit();

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

        if (static_cast<int>(gamepad_.Y.pressed) == 1)
        {
          if (command_speed < 0.03)
          {
            std::cout << "[INFO] Y button pressed!" << std::endl;
            state_ = State::RL_POLICY_WAVE_HAND;            
          }
        }

        rl_action_ = RunInference();

        mode_pr_ = Mode::PR;

        for (size_t i = 0; i < cfg.num_actions; ++i) // 12
        {
          int i2m_idx = isaaclab2mujoco[i];
          motor_command_tmp.q_target[i] = (rl_action_.at(i2m_idx) * cfg.action_scale) + cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.rl_kp[i];
          motor_command_tmp.kd.at(i) = cfg.rl_kd[i];
        }

        // upper: keep default pose
        for (size_t i = cfg.num_actions; i < G1_NUM_MOTOR; ++i)
        {
          motor_command_tmp.q_target[i] = cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.init_kp[i];
          motor_command_tmp.kd.at(i) = cfg.init_kd[i];
        }

        // upper: arm swing motion
        // Quat2RPY(); 500Hz
        // cout << arm_swing_motion.at(0) << endl;

        // get velocity command
        motor_command_tmp.q_target[15] = arm_swing_motion.at(0) + cfg.default_pos[15];
        motor_command_tmp.q_target[22] = arm_swing_motion.at(1) + cfg.default_pos[22];

        motor_command_tmp.q_target[18] = arm_swing_motion.at(0) + cfg.default_pos[18];
        motor_command_tmp.q_target[25] = arm_swing_motion.at(1) + cfg.default_pos[25];

        // float x_vel = static_cast<float>(gamepad_.ly) * cfg.cmd_scale[0] * cfg.max_cmd[0];
        // motor_command_tmp.q_target[15] = x_vel * arm_swing_motion.at(0) + cfg.default_pos[15];
        // motor_command_tmp.q_target[22] = x_vel * arm_swing_motion.at(1) + cfg.default_pos[22];

        // motor_command_tmp.q_target[18] = x_vel * arm_swing_motion.at(0) + cfg.default_pos[18];
        // motor_command_tmp.q_target[25] = x_vel * arm_swing_motion.at(1) + cfg.default_pos[25];

        // float y_vel = 1.4 * fabsf(static_cast<float>(gamepad_.lx) * -1 * cfg.cmd_scale[1] * cfg.max_cmd[1]);
        // motor_command_tmp.q_target[16] = -y_vel * arm_swing_motion.at(0) + cfg.default_pos[16];
        // motor_command_tmp.q_target[23] = y_vel * arm_swing_motion.at(1) + cfg.default_pos[23];

        CheckExit();

        break;
      }

      case State::RL_POLICY_WAVE_HAND:
      {
        // [Stage 4]: run robot with RL policy & wave hand
        time_ += command_dt_;

        if (static_cast<int>(gamepad_.select.pressed) == 1)
        {
          std::cout << "[INFO] select button pressed!" << std::endl;
          state_ = State::DAMPING_STATE;
        }
        
        rl_action_ = RunInference();

        mode_pr_ = Mode::PR;

        // lower: RL_POLICY_ACTIVE
        for (size_t i = 0; i < cfg.num_actions; ++i) // 12
        {
          int i2m_idx = isaaclab2mujoco[i];
          motor_command_tmp.q_target[i] = (rl_action_.at(i2m_idx) * cfg.action_scale) + cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.rl_kp[i];
          motor_command_tmp.kd.at(i) = cfg.rl_kd[i];
        }

        // upper: keep default pose
        for (size_t i = cfg.num_actions; i < G1_NUM_MOTOR; ++i)
        {
          motor_command_tmp.q_target[i] = cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.init_kp[i];
          motor_command_tmp.kd.at(i) = cfg.init_kd[i];
        }

        // upper: WAVE_HAND
        if (time_ < duration_) 
        {
          for (size_t i = 22; i < 26; ++i) 
          {
            double ratio = std::clamp(time_ / duration_, 0.f, 1.f);
            double smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio);
            motor_command_tmp.q_target.at(i) = (1.0 - smooth_ratio) * ms->q.at(i) + smooth_ratio * arm_swing_wave_hand_goal[i-22];
            motor_command_tmp.kp.at(i) = cfg.rl_kp[i];
            motor_command_tmp.kd.at(i) = cfg.rl_kd[i];
          }
        }
        else if (time_ < duration_ + wave_duration) // wave_hand & waist yaw
        {
          double t_sine = time_ - duration_;

          for (size_t i = 22; i < 26; ++i) 
          {
            motor_command_tmp.q_target.at(i) = arm_swing_wave_hand_goal[i - 22];
            motor_command_tmp.kp.at(i) = cfg.rl_kp[i];
            motor_command_tmp.kd.at(i) = cfg.rl_kd[i];
          }

          motor_command_tmp.q_target.at(12) = amp * std::sin(omega * t_sine / 3);
          motor_command_tmp.kp.at(12) = cfg.rl_kp[12];
          motor_command_tmp.kd.at(12) = cfg.rl_kd[12];

          motor_command_tmp.q_target.at(23) = arm_swing_wave_hand_goal[1] + amp * (1.0 - std::cos(omega * t_sine));
          motor_command_tmp.q_target.at(25) = arm_swing_wave_hand_goal[3] + (amp * 0.5) * (1.0 - std::cos(omega * t_sine));
        }
        else if (time_ < duration_ + wave_duration + return_duration) // move to default pose
        {
          double ratio = std::clamp((time_ - (duration_ + wave_duration))/ return_duration, 0.f, 1.f);
          double smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio);

          for (size_t i = 15; i < 29; ++i)
          {
            motor_command_tmp.q_target.at(i) = (1.0 - ratio) * ms->q.at(i) + ratio * cfg.default_pos[i];
            motor_command_tmp.kp.at(i) = cfg.rl_kp[i];
            motor_command_tmp.kd.at(i) = cfg.rl_kd[i];
          }

          motor_command_tmp.q_target.at(12) = (1.0 - smooth_ratio) * ms->q.at(12) + smooth_ratio * cfg.default_pos[12];
          motor_command_tmp.kp.at(12) = cfg.rl_kp[12];
          motor_command_tmp.kd.at(12) = cfg.rl_kd[12];
        }
        else
        {
          if (static_cast<int>(gamepad_.A.pressed) == 1)
          {
            time_ = 0.0;
            std::cout << "[INFO] A button pressed! RL_POLICY_ACTIVE" << std::endl;
            Global2NewWorld();
            state_ = State::RL_POLICY_ACTIVE;
          }

          // Not Recommended --------------------------------------------------------------------
          if (static_cast<int>(gamepad_.X.pressed) == 1)
          {
            time_ = 0.0;
            std::cout << "[INFO] X button pressed! WAIT_FOR_POLICY_COMMAND" << std::endl;
            Global2NewWorld();
            state_ = State::WAIT_FOR_POLICY_COMMAND;
          }
        }

        CheckExit();

        break;
      }

      case State::DAMPING_STATE:
      {
        // [Stage 5]: finish robot control
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
    // std::cerr << "[WARNING] MotorState not available!\n";
  }

  motor_command_buffer_.SetData(motor_command_tmp);
}

int main()
{
  WholeBodyRL wholebodyrl("eth0", "../configs/g1_arc.yaml");
  // WholeBodyRL wholebodyrl("wlo1", "../configs/g1_arc.yaml");

  while (true)
  {
    if (wholebodyrl.should_exit_) break;
    std::this_thread::sleep_for(std::chrono::microseconds(2000));  // 2ms = 2000μs
  }

  return 0;
}