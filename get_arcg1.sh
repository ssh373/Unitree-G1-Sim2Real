#!/bin/bash

source ~/.bashrc
clear

echo "/*******************************************************************************************"
echo "* "
echo "*  HumARConoid-Sim2Real"
echo "* "
echo "*  Sim2Real Transfer to Optimize Humanoid Locomotion Strategy using Reinforcement Learning"
echo "* "
echo -e "*    \e[4mhttps://github.com/ARC-KIST/arc_g1.git\e[0m"
echo "* "
echo "*  Advanced Robot Control Lab. (ARC)"
echo "*      @ Korea Institute of Science and Technology"
echo "* "
echo -e "*    \e[4mhttps://sites.google.com/view/kist-arc\e[0m"
echo "* "
echo "*******************************************************************************************/"
echo ""
echo "/* Author: Sol Choi (Jennifer) */"
echo ""


rsync -avz --exclude 'onnxruntime' --exclude 'yaml-cpp' unitree@192.168.123.164:/home/unitree/arc_ws/arc_g1/ /home/kist/arc_backup
