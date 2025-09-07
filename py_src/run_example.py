import time
import numpy as np
import torch
from py_src.policy import ARC_G1_POLICY_DIR

class Controller():
    def __init__(self):
        self.dt = 0.02
        self.counter = 0
        self.start_time = time.perf_counter()
        self.next_time = time.perf_counter()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs = np.zeros(47, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self.obs_torch = torch.zeros(47, dtype=torch.float32)

        self.target_dof_pos = np.zeros(29, dtype=np.float32)
        self.buffer_time = []
        self.buffer_cur_pos = []
        self.buffer_des_pos = []
        self.buffer_timimi = []

        self.policy = torch.jit.load(f"{ARC_G1_POLICY_DIR}/g1_pretrained/motion.pt").to(self.device)
        

    def run(self):
        run_time = time.perf_counter()
        if self.counter == 0:
            self.next_time = run_time
        self.counter += 1

        period = 0.8
        count = self.counter * 0.02
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)     

        qj_obs = np.zeros(29, dtype=np.float32)
        # data logging (.npz)
        self.buffer_cur_pos.append(qj_obs)

        target_dof_pos_log = self.target_dof_pos.flatten()
        self.buffer_des_pos.append(target_dof_pos_log)

        self.target_dof_pos[0] = 0.5 * sin_phase - 0.1

        # Prepare observation tensor and move it to GPU
        self.obs[:3] = [0, 0, 0]
        self.obs[3:6] = [0, 0, -1]
        self.obs[6:9] = [0, 0, 0]
        self.obs[9 : 9 + 12] = np.zeros(12, dtype=np.float32)
        self.obs[9 + 12 : 9 + 12 * 2] = np.zeros(12, dtype=np.float32)
        self.obs[9 + 12 * 2 : 9 + 12 * 3] = np.zeros(12, dtype=np.float32)
        
        self.obs_torch[:3] = [0, 0, 0]
        self.obs_torch[3:6] = [0, 0, -1]
        self.obs_torch[6:9] = [0, 0, 0]
        self.obs_torch[9 : 9 + 12] = torch.zeros(12, dtype=np.float32)
        self.obs_torch[9 + 12 : 9 + 12 * 2] = torch.zeros(12, dtype=np.float32)
        self.obs_torch[9 + 12 * 2 : 9 + 12 * 3] = torch.zeros(12, dtype=np.float32)

        # Get the action from the policy network
        start_inf = time.perf_counter()
        obs_tensor = torch.from_numpy(self.obs).to(self.device).unsqueeze(0)
        # obs_tensor = torch.tensor(self.obs).pin_memory().to(self.device).unsqueeze(0)
        # obs_tensor = torch.tensor(self.obs, dtype=torch.float32).pin_memory().to(self.device).unsqueeze(0)

        self.action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()
        end_inf = time.perf_counter()

        self.next_time += self.dt
        sleep_time = self.next_time - time.perf_counter()

        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Loop overran by {-sleep_time:.6f} sec, next_time: {self.next_time}")
            self.next_time = time.perf_counter()

        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time

        self.buffer_time.append(elapsed_time)
        self.buffer_timimi.append(end_inf - start_inf)


if __name__ == "__main__":

    controller = Controller()

    dummy_input = torch.zeros(1, 47).to(controller.device)
    controller.policy(dummy_input)

    controller.start_time = time.perf_counter()
    for _ in range(1000):
        try:
            controller.run()
        except KeyboardInterrupt:
            break

    controller.cur_pos_array = np.array(controller.buffer_cur_pos)
    controller.des_pos_array = np.array(controller.buffer_des_pos)
    controller.time_array = np.array(controller.buffer_time)
    controller.timimi_array = np.array(controller.buffer_timimi)

    np.savez('../log_data_simple_desktop.npz', cur_pos=controller.cur_pos_array, des_pos=controller.des_pos_array, time=controller.time_array, dt=controller.timimi_array)

    print("Exit")
