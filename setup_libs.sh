#!/bin/bash

source ~/.bashrc
clear

echo "/**************************************************************"
echo "* "
echo "*  Advanced Robot Control Lab. (ARC)"
echo "*      @ Korea Institute of Science and Technology"
echo "* "
echo -e "*    \e[4mhttps://sites.google.com/view/kist-arc\e[0m"
echo "* "
echo "**************************************************************/"
echo ""
echo "/* Author: Sol Choi (Jennifer) */"
echo ""

echo -e "\e[32m[INFO] Set up Unitree G1 Development package START!\e[0m"

pwd=$(pwd)

unitree_sdk2_python_path="$HOME/unitree_sdk2_python"

if [ -d "$unitree_sdk2_python_path" ]; then
    echo -e "[INFO] unitree_sdk2_python exists."
else
    echo -e "\e[33m[WARN] unitree_sdk2_python does not exists. Try to install...\e[0m"
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
fi

set -e

# apt install python3-pip
cd $unitree_sdk2_python_path
pip install -e .

set -e

echo -e "\e[32m[INFO] 'unitree_sdk2_python' successfully installed!\e[0m"
echo -e "\e[0m[INFO] Try to install arc_g1 library...\e[0m"

cd $HOME
cd $pwd

pip install -e .

set -e 
echo -e "\e[32m[INFO] 'arc_g1' successfully installed!\e[0m"


# robot network interface name
# ip a | awk '/inet 192.168.123.164/ {print $NF}'