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

## Install

**Step 1.** Install `unitree_sdk2`

Check out the detailed installation guide of [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) in this page
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
**Step 1.** Make sure the path: *unitree_sdk2*  
in `CMakeLists.txt`,
```
# ==== PATHS ====
set(UNITREE_SDK2_DIR ${CMAKE_SOURCE_DIR}/../unitree_sdk2) # change to your path
```
- The `UNITREE_SDK2_DIR` value must point to your actual unitree_sdk2 install path.

**Step 2.** Build the project

> Check your robot's [NETWORK_CARD_NAME](https://support.unitree.com/home/en/G1_developer/quick_development#:~:text=subnet%20using%20the-,ifconfig,-command%2C%20as%20shown)!  
> (for simple check, use `ifconfig` in terminal)

```
mkdir build && cd build
cmake .. && make -j$(nproc)

# Run the arm example (7-DoF arm) from the build directory
./g1_arm_example NETWORK_CARD_NAME
```

**Step 3.** Run the Sim2Real Controller
>[!Tip]  
> Change the `configs/g1_sim2real.yaml` file for your settings
```
./wholebody_rl # run from the build directory
```

**Step 4.** Enjoy your experiments!
| button | status |
|:--------:|:------:|
| start       | move robot to the default pose      |
| A       | RL policy active      |
| select       | damping mode & kill      |

- Step 1.
    Press __start__ button to set the robot to the default position!
- Step 2.
    Place the robot on the floor! (Robot must be standing upright)
- Step 3.
    Press __A__ to run the RL policy (Sim2Real deploy)

> [!Caution]  
> User should always prepare to press the __select__ button!  
> (it works like kill button, but do not rely on it!)

</br>

### License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

</br>

### References
- [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)

</br></br>

**Last update**: _26.01.05_ [_Sol Choi_](https://github.com/S-CHOI-S)
