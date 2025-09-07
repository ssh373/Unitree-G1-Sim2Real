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

# check conda is installed
source $HOME/*conda*/etc/profile.d/conda.sh
pwd=$(pwd)

if ! command -v conda &> /dev/null
then
    echo -e "\e[31m[ERROR] Conda could not be found. Please install conda and try again.\e[0m"
    exit 1
fi

echo -e "\e[32m[INFO] Set up Unitree G1 conda environment START!\e[0m"
echo ""
read -p $'\e[0m>> Conda environment name to be created: \e[0m' env_name
read -p "$(echo -e '\e[0m>> New conda environment name: \e[33m'"$env_name"'\e[0m, start to create env? (y/n) \e[0m')" create_env

set -e

create_env=${create_env:-y}

if [ "$create_env" = "y" ] || [ "$create_env" = "yes" ]; then
    if conda env list | grep -w "$env_name"; then
        read -p "$(echo -e "\e[31m[ERROR] Conda environment named '${env_name}' already exists.\e[0m Do you want to continue? (y/n)")" continue
        continue=${continue:-y}
        if ! [ "$continue" = "y" ] || [ "$continue" = "yes" ]; then
            echo -e "\e[31m[ERROR] Exiting without creating the conda environment."
            exit 0
        fi
    else
        echo -e "[INFO] Creating conda environment named '${env_name}'..."
        conda env create --name ${env_name} --file environment.yaml
    fi
else
    echo -e "\e[31m[ERROR] Exiting without creating the conda environment."
    exit 0
fi

set -e

source ~/.bashrc
source $HOME/*conda*/etc/profile.d/conda.sh
conda activate ${env_name}
set -e

if ["$CONDA_DEFAULT_ENV" = "$env_name"]; then
    echo -e "\e[32m[INFO] Successfully activated '$env_name' env!\e[0m"
else
    echo -e "\e[31m[ERROR] '$env_name' env is deactivated.\e[0m"
    echo -e "\e[0m[INFO] Please activate the \e[33m'${env_name}'\e[0m environment and start to develop!"
    exit 0
fi