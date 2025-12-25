# Unitree-G1 Sim2Real Depoly Extension Template
> [!note]  
> This extension template is for `unitree_sdk2`!  
> **Author**: [_Sol Choi_](https://github.com/S-CHOI-S)

![alt text](./docs/unitree_g1.png)


## Install

**Step 1.** Install Unitree-SDK2

Checkout the detailed installation guide of [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) in this page
```

```

**Step 1.** Clone the repository

```
git clone --recursive  https://github.com/S-CHOI-S/Unitree-G1-Sim2Real.git g1_sim2real
```

submodule init
submodule update

**Step 2.** build onnxruntime

```
./arc_g1.sh  # create conda env
git checkout v1.11.0
./build.sh --config Release --build_shared_lib --parallel --update --build --build_dir build --enable_pybind --disable_ml_ops
```

**Step 3.** build yaml-cpp
```
cmake .. -DYAML_BUILD_SHARED_LIBS=ON
```
</br>

## Usage
### C++ Development
```
mkdir build && cd build
```
```
cmake .. && make -j$(nproc)
```
```
./g1_arm_example # run from the build directory
```
</br>

### Python Development
> [!note]  
> Please make sure that `./setup_libs.sh` has been executed.

```
cd py_src
```
```
python check_robot_config.py
```
</br>

## Default Settings
> [!Tip]  
> Onboard computer: __*unitree@192.168.123.164*__  
> Python version: **python3.10**  
> ROS version: **(1) foxy, (2) noetic**  
> G1 network card name: __*eth0*__

![alt text](./docs/transition_operationg_mode.png)

</br>

## Development Settings
> If you are an ARC member, please refer to **[this](https://www.notion.so/G1-19f147ce6766809cafd8dc87954b44ec?pvs=4)** page!

### References
https://www.unitree.com/g1/  
https://www.unitree.com/app/g1/  
https://support.unitree.com/main  
https://support.unitree.com/home/en/G1_developer/about_G1  
https://www.docs.quadruped.de/projects/g1/html/index.html  

</br></br>

**Last update**: _25.12.25_ [_Sol Choi_](https://github.com/S-CHOI-S)
