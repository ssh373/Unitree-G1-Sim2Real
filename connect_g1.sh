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


sudo ifconfig eth0 down

set -e

sudo ifconfig eth0 192.168.123.162/164

set -e

sudo ifconfig eth0 up

set -e

echo -e "[INFO] Successfully set network as '\e[32meth0: 192.168.123.162/164\e[0m'!\e[0m"

set -e
