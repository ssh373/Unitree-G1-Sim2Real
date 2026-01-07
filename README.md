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
git submodule update --init --recursive
```

**Step 3.** Build submodule #1: `onnxruntime`

```
cd thirdparty/onnxruntime
./build.sh --config Release --build_shared_lib --parallel --update --build --build_dir build --enable_pybind --disable_ml_ops
```

**Step 4.** Build submodule #2: `yaml-cpp`
```
cd thirdparty/yaml-cpp
mkdir build && cd build
cmake .. -DYAML_BUILD_SHARED_LIBS=ON
make -j$(nproc)
```
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

**Step 3.** Run the Sim2Real Controller
>[!Note]  
> Modify `configs/g1_sim2real.yaml` file for your settings
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
The code is structured as follows:
- `configs/g1_sim2real.yaml` — main configuration (network settings, robot params, policy params)
- `src/wholebody_rl.cpp` — communicate with the robot and execute the trained policy
- `policy/` — trained policy files (.onnx)

To adapt to your own policy, modify:
1. Replace the `.onnx` file path in `configs/g1_sim2real.yaml` with your trained policy
2. Adjust `include/g1_sim2real/wholebody_rl.hpp` and `src/wholebody_rl.cpp` to match your policy's observation/action space and terms

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