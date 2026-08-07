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

WholeBodyRL::WholeBodyRL(const std::string& config_yaml_path)
: mode_pr_(Mode::PR),
  time_(0.0),
  time_abs(0.0),
  control_dt_(CONTROL_DT), // 500Hz
  command_dt_(COMMAND_DT), // 50Hz
  duration_(5.0) // move to default pose
{
  LoadYamlConfig(config_yaml_path);
  LoadOnnxModel();

  logger_thread_ = std::thread(&WholeBodyRL::LoggerThread, this);
}

WholeBodyRL::~WholeBodyRL()
{
  should_exit_ = true;
  logging_active_ = false;

  if (logger_thread_.joinable()) logger_thread_.join();
}


/*****************************************************************************
** Comm / sim seam
*****************************************************************************/
void WholeBodyRL::read(float dt, const MotorState& ms, const ImuState& is)
{
  control_dt_ = dt;

  motor_state_buffer_.SetData(ms);
  imu_state_buffer_.SetData(is);
}

void WholeBodyRL::set_input(const CommandInput& in)
{
  input_ = in;
}

bool WholeBodyRL::write(MotorCommand& mc)
{
  const std::shared_ptr<const MotorCommand> latest = motor_command_buffer_.GetData();
  if (!latest) return false;

  mc = *latest;
  return true;
}

void WholeBodyRL::control()
{
  // The policy runs at 50Hz, the PD command at 500Hz: the same split the two
  // recurrent DDS threads used to provide.
  if (step_cnt % DECIMATION == 0) Control();
  ++step_cnt;

  CheckSafetyLimits();

  // Phase time base. Must tick at 1/control_dt_ or the gait period drifts.
  if (state_ == State::RL_POLICY_ACTIVE) cnt++;
}

void WholeBodyRL::Reset()
{
  state_ = State::WAIT_FOR_INIT_COMMAND;
  mode_pr_ = Mode::PR;

  time_ = 0.0f;
  time_abs = 0.0f;
  cnt = 0;
  step_cnt = 0;

  rl_action_ = {};
  input_ = CommandInput{};

  logging_active_ = false;
  should_exit_ = false;

  motor_command_buffer_.Clear();
  motor_state_buffer_.Clear();
  imu_state_buffer_.Clear();

  std::cout << "[INFO] Controller reset." << std::endl;
}


/*****************************************************************************
** Thread functions
*****************************************************************************/
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


/*****************************************************************************
** Config & Model loading functions
*****************************************************************************/
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
    std::vector<float> cmd_scale = config["cmd_scale"].as<std::vector<float>>();
    std::copy(cmd_scale.begin(), cmd_scale.end(), cfg.cmd_scale.begin());
    std::vector<float> max_cmd = config["max_cmd"].as<std::vector<float>>();
    std::copy(max_cmd.begin(), max_cmd.end(), cfg.max_cmd.begin());
    
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
  std::cout << "    max_cmd        : [" << cfg.max_cmd[0] << ", " << cfg.max_cmd[1] << ", " << cfg.max_cmd[2] << "]\n";

  std::cout << "  Model dimensions:\n";
  std::cout << "    num_obs        : " << NUM_OBS<< "\n";
  std::cout << "    num_actions    : " << NUM_ACTIONS << "\n";
  std::cout << "================================================================================\n" << std::endl;
}

void WholeBodyRL::LoadOnnxModel()
{
  // initialize onnx runtime env
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "wholebody_rl");

  // initialize input data
  input_data = std::vector<float>(NUM_OBS, 0.0f);
  input_shape = {1, NUM_OBS};

  // session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // load onnx model
  session_ = std::make_unique<Ort::Session>(env_, cfg.policy_path.c_str(), session_options);

  // Prepare input/output names
  // The *Allocated variants hand back ownership of the string, so copy it into a
  // member that outlives input_names/output_names.
  input_name_holder_ = session_->GetInputNameAllocated(0, allocator_).get();
  output_name_holder_ = session_->GetOutputNameAllocated(0, allocator_).get();

  // input/output names
  input_names = {input_name_holder_.c_str()};
  output_names = {output_name_holder_.c_str()};
}


/*****************************************************************************
** Control functions
*****************************************************************************/
void WholeBodyRL::Control() 
{
  // monitor control freq monitor
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

    switch (state_) 
    {
      case State::WAIT_FOR_INIT_COMMAND:
      {
        // [Stage 0]: set robot to wait for init command
        if (input_.start)
        {
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

        if (input_.policy)
        {
          time_ = 0.0;
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
        if (input_.damping)
        {
          std::cout << "[INFO] damping requested!" << std::endl;
          state_ = State::DAMPING_STATE;
        }

        rl_action_ = RunInference();

        mode_pr_ = Mode::PR;

        for (size_t i = 0; i < NUM_ACTIONS; ++i)
        {
          int i2m_idx = isaaclab2mujoco[i];
          motor_command_tmp.q_target[i] = (rl_action_.at(i2m_idx) * cfg.action_scale) + cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.rl_kp[i];
          motor_command_tmp.kd.at(i) = cfg.rl_kd[i];
        }

        // upper: keep default pose
        for (size_t i = NUM_ACTIONS; i < G1_NUM_MOTOR; ++i)
        {
          motor_command_tmp.q_target[i] = cfg.default_pos[i];
          motor_command_tmp.kp.at(i) = cfg.init_kp[i];
          motor_command_tmp.kd.at(i) = cfg.init_kd[i];
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
      if (ms->q.at(i) < joint_pos_min.at(i) // check joint position limit (min)
          || ms->q.at(i) > joint_pos_max.at(i) // check joint position limit (max)
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
std::vector<float> WholeBodyRL::GetObservation()
{
  const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
  const std::shared_ptr<const ImuState> is = imu_state_buffer_.GetData();
  
  if (ms && is)
  {
    std::vector<float> obs(NUM_OBS);
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
    // Sign conventions live in the comm layer: it maps its own input device onto
    // these three velocities (see UnitreeComm::LowStateHandler).
    obs[6] = input_.lin_vel_x * cfg.cmd_scale[0] * cfg.max_cmd[0];
    obs[7] = input_.lin_vel_y * cfg.cmd_scale[1] * cfg.max_cmd[1];
    obs[8] = input_.ang_vel_z * cfg.cmd_scale[2] * cfg.max_cmd[2];

    // joint pos
    for (size_t i = 0; i < NUM_ACTIONS; ++i) // NUM_ACTIONS
    {
      int m2i_idx = mujoco2isaaclab[i];
      obs[start_idx + i] = (ms->q[m2i_idx] - cfg.default_pos[m2i_idx]) * cfg.dof_pos_scale;
    }

    // joint vel
    start_idx += NUM_ACTIONS;
    for (size_t i = 0; i < NUM_ACTIONS; ++i) // NUM_ACTIONS
    {
      int m2i_idx = mujoco2isaaclab[i];
      obs[start_idx + i] = ms->dq[m2i_idx] * cfg.dof_vel_scale;
    }

    // action
    std::copy(rl_action_.begin(), rl_action_.end(), obs.begin() + start_idx + NUM_ACTIONS); // NUM_ACTIONS

    float period = 0.8f;
    float phase = std::fmod(cnt * control_dt_, period) / period;

    // sin phase
    obs[45] = std::sin(2.0 * M_PI * phase);

    // cos phase
    obs[46] = std::cos(2.0 * M_PI * phase);

    float cmd_speed = std::sqrt(obs[6] * obs[6] + obs[7] * obs[7] + obs[8] * obs[8]);
    
    if (cmd_speed < 0.1f)
    {
      obs[45] = 0.0f;
      obs[46] = 0.0f;
    }

    start_idx = 9;

    return obs;
  }

  if (!ms || !is) 
  {
    std::cerr << "[ERROR] Failed to make Observation! Change to Damping mode!" << std::endl;
    state_ = State::DAMPING_STATE;
    should_exit_=true;
  }

  return std::vector<float>(NUM_OBS, 0.0f);
}

std::array<float, NUM_ACTIONS> WholeBodyRL::RunInference()
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
    std::cerr << "Inference failed: " << e.what() << std::endl;
  }

  // process the output tensor
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

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



