# Unitree-G1 Sim2Real Deploy Extension Template
> [!note]  
> Minimal Sim2Real deploy template to run a trained RL policy on the real Unitree G1.  
> **Author**: [_Sol Choi_](https://github.com/S-CHOI-S)

![alt text](https://www.unitree.com/images/52688de58de044358e4792a5b7c1593d_2740x1720.jpg)

</br>

## Tested Version
- OS: *Ubuntu 20.04/22.04*
- Unitree G1 Model Number: *5, 15*

</br>

## Installation

**Step 1.** Install `unitree_sdk2`

Check out the detailed installation guide of [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) on this page.
```
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Step 2.** Clone the `g1_sim2real` repository

```
git clone --recursive  https://github.com/S-CHOI-S/Unitree-G1-Sim2Real.git g1_sim2real
```
- `--recursive` now pulls only `yaml-cpp` (~6 MB). Dropping it is fine too — see Step 3.

**Step 3.** Third-party dependencies — nothing to do

No manual dependency build is required. CMake resolves everything during `cmake ..`:

| Dependency | How it is resolved | Notes |
|---|---|---|
| `onnxruntime` | Prebuilt release tarball unpacked into `thirdparty/` | CPU build, `x64` / `aarch64` selected automatically |
| | | Requires **1.12 or newer**; override with `-DONNXRUNTIME_VERSION=` |
| `yaml-cpp` | Vendored source → system `libyaml-cpp-dev` → source download | No official prebuilt exists upstream; linked statically |
| `mujoco` | Prebuilt release tarball, **opt-in** | `cmake .. -DG1_WITH_MUJOCO=ON`; needs `libglfw3-dev` |

The download happens **once**, on the first `cmake ..`, and lands in `thirdparty/`:

```
thirdparty/
├── onnxruntime-linux-x64-1.28.0/   ← 25 MB, downloaded on first configure
├── mujoco-3.11.0/                  ← 85 MB, only with -DG1_WITH_MUJOCO=ON
└── yaml-cpp/                       ← submodule
```

Deleting `build/` does **not** trigger a re-download — only removing the `thirdparty/` directory does.
The directory name carries the version and architecture, so bumping a version fetches a new one
instead of silently reusing the old one. Both are covered by `.gitignore`.

Pinned versions live at the top of `CMakeLists.txt` and are SHA256-verified:

```
set(ONNXRUNTIME_VERSION 1.28.0 CACHE STRING "...")
set(MUJOCO_VERSION      3.11.0 CACHE STRING "...")
```

<details>
<summary>Building on a machine without internet access (e.g. the onboard PC)</summary>

Unpack the tarball into `thirdparty/` yourself — the directory name CMake looks for is exactly
the one inside the archive, so there is nothing else to configure:

```
# on a machine with internet (use aarch64 for the onboard Jetson)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-aarch64-1.28.0.tgz

# on the target machine
tar xzf onnxruntime-linux-aarch64-1.28.0.tgz -C thirdparty/
cmake ..     # finds thirdparty/onnxruntime-linux-aarch64-1.28.0, downloads nothing
```

To keep it somewhere else entirely, pass the path instead:

```
cmake .. -DONNXRUNTIME_DIR=/opt/onnxruntime-linux-aarch64-1.28.0
```

MuJoCo works the same way via `thirdparty/mujoco-<version>/` or `-DMUJOCO_DIR=`.
</details>

</br>

## Usage
### Experiment
**Step 1.** Set the `unitree_sdk2` path in `CMakeLists.txt` 
```
# ==== PATHS ====
set(UNITREE_SDK2_DIR ${CMAKE_SOURCE_DIR}/../unitree_sdk2) # change to your path
```
- The `UNITREE_SDK2_DIR` value must point to your actual `unitree_sdk2` install path.


**Step 2.** Build the project

> Check your robot's [NETWORK_CARD_NAME](https://support.unitree.com/home/en/G1_developer/quick_development#:~:text=subnet%20using%20the-,ifconfig,-command%2C%20as%20shown)!  
> (for a simple check, use `ifconfig` in terminal)

```
mkdir build && cd build
cmake .. && make -j$(nproc)

# Run the arm example (7-DoF arm) from the build directory
./g1_arm_example NETWORK_CARD_NAME
```
- The first `cmake ..` downloads the `onnxruntime` prebuilt binary into `thirdparty/` (~9 MB archive). Every later configure reuses it, including after `rm -rf build`.
- `wholebody_rl` is linked with `RPATH=$ORIGIN`, so copying the executable together with `libonnxruntime.so*` into one directory is enough to run it on the robot — no `LD_LIBRARY_PATH` needed.

**Step 3.** Run the Sim2Real Controller
>[!Note]  
> Modify `configs/g1_sim2real.yaml` file for your settings


>[!Warning]  
> The current policy exhibits high impact forces. This will be replaced with an improved policy in a future update.

```
./wholebody_rl # run from the build directory
```

**Step 4.** Enjoy your experiments!

| Button | Function |
|:------:|:---------:|
| Start | Move robot to the default pose |
| A | Activate RL policy |
| Select | Damping mode & kill |

| Control | Action |
|:-------:|:-------:|
| Left Rocker ↕ | Move forward/backward (±X) |
| Left Rocker ↔ | Move sideways (±Y) |
| Right Rocker ↔ | Rotate left/right (±Yaw) |

**How to run:**
1. Press **Start** to set the robot to the default position.
2. Place the robot on the floor. (Robot must be standing upright)
3. Press **A** to run the RL policy (Sim2Real deploy).
4. Use the rockers to control movement.

> [!Caution]  
> Always be ready to press the **Select** button!  
> (It works as a kill button, but do not rely on it!)

**Step 5.** Check experiment logs
- Use `src/plot.ipynb` to visualize your logs.

</br>

### Customization

The control logic lives in one place and knows nothing about how it is connected. Two thin
layers put it either on the robot or in the simulator:

| | |
|---|---|
| `src/wholebody_rl.cpp` | control: state machine, observation, ONNX inference |
| `src/unitree_comm.cpp` | real robot: DDS, CRC, gamepad, 500 Hz loop |
| `src/mujoco_sim.cpp` | simulation: `mj_step`, PD in place of the motor driver |
| `src/mujoco_viewer.cpp` | simulation: GLFW window, camera, keyboard |
| `include/g1_sim2real/robot_types.hpp` | the vocabulary all of them agree on |
| `configs/g1_sim2real.yaml` | network, robot and policy parameters |
| `policy/` | trained policy files (.onnx) |
| `model/` | MuJoCo model used by the simulation |
| `tools/policy_check.cpp` | offline inference dump, see below |
| `tools/fake_lowstate.cpp` | stands in for the robot on loopback, see below |

They meet at one seam:

```
                   read(dt, MotorState, ImuState)
  UnitreeComm  ───────────────────────────────────▶  WholeBodyRL
      or       ─── set_input(CommandInput) ───────▶   state machine,
  MujocoSim    ◀── write(MotorCommand) ────────────   observation, ONNX
                 control()  ← the caller owns the loop
```

`MotorCommand` carries `q_target / kp / kd / tau_ff`. On the robot the motor driver closes
that PD loop itself; in MuJoCo `MujocoSim` closes it and writes torque to `d->ctrl`. That is
essentially the whole difference between the two.

### Running in simulation (sim2sim)

```
cmake .. -DG1_WITH_MUJOCO=ON && make -j$(nproc)
./wholebody_sim [config.yaml] [model.xml] [duration_s] [vx] [vy] [wz]
```

`duration_s` of 0 (the default) runs until the window is closed. `vx vy wz` seed the command;
the keyboard trims it from there:

| | |
|---|---|
| up / down (or `W` `S`) | walk forward / back |
| left / right (or `Q` `E`) | turn |
| `A` / `D` | step sideways |
| `shift` | slower trim, for dialling a value in |
| `X` | zero the command |
| `space` / `backspace` | pause / restart the episode |
| `R` / `esc` | reset the camera / quit |

The command is a trim rather than a spring-loaded stick: a held key moves it and releasing
leaves it there, so any value in `-1..1` is reachable. It is normalised — `cmd_scale * max_cmd`
in the YAML turns it into m/s and rad/s, so `1.0` is 1.0 m/s forward by default.

> [!Note]
> The floating base is pinned for the first 7 s. Until the policy takes over, the controller
> only closes a joint-space PD loop — there is no balance control, because on hardware the
> robot is suspended or held while it moves to the default pose. Without pinning it simply
> topples during those seconds and the policy inherits a robot lying face down.

To adapt to your own policy, modify:
1. Replace the `.onnx` file path in `configs/g1_sim2real.yaml` with your trained policy
2. Adjust `include/g1_sim2real/wholebody_rl.hpp` and `src/wholebody_rl.cpp` to match your policy's observation/action space and terms

</br>

### Checking a policy offline

`policy_check` loads a `.onnx` with the same session setup as `wholebody_rl` and dumps the
outputs for three fixed observation vectors — including the raw float bit patterns, since two
`onnxruntime` versions rarely agree bit-for-bit (the Gemm kernels accumulate in a different order).

```
./policy_check ../policy/basic_walk/policy.onnx   # run from the build directory
```
```
onnxruntime : header ORT_API_VERSION=28, runtime 1.28.0
input       : obs [1, 47]
output      : actions [1, 12]
case0 bits  : bedbeefd bcb51ba8 3ddd698e ...
case0 val   : -0.429557711 -0.0221079141  0.108111486 ...
```

Diff two runs to see whether anything actually changed after bumping `ONNXRUNTIME_VERSION` or
re-exporting a policy. Shapes are read from the model, so it works with any policy.
Disable the tool with `cmake .. -DG1_BUILD_TOOLS=OFF`.

</br>

---

### Contact
For questions or issues:
- Open a [GitHub Issue](https://github.com/S-CHOI-S/Unitree-G1-Sim2Real/issues)
- Email: `solchoi@yonsei.ac.kr` or `solchoi@kist.re.kr`

### License
This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.


### References
- [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)

</br>

**Last update**: _2026.01.07_ [_Sol Choi_](https://github.com/S-CHOI-S)