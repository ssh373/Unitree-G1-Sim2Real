# ARC Humanoid Robot G1 Development Extension Template
> [!note]  
> This extension template is for `unitree_sdk2` and `unitree_sdk2_python`!  
> **Author**: [_Sol Choi_](https://github.com/S-CHOI-S)

![alt text](./docs/unitree_g1.png)


## Install
**Step 1.** clone the repository

```
git clone --recursive -b sol_sim2real https://github.com/ARC-KIST/arc_g1.git your_repository_name
```

**Step 2.** create conda environment

```
./arc_g1.sh  # create conda env
```
```
conda activate your_env_name
```

**Step 3.** install dependecies
```
./setup_libs.sh  # install dependencies
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

**Last update**: _25.05.09_ [_Sol Choi_](https://github.com/S-CHOI-S)
