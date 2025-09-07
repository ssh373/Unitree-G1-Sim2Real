"""******************************************************************************************
* HumARConoid-Mujoco
*
* Sim2Sim Transfer to Optimize Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid-MuJoCo.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

import os
import toml

##
# Configuration for different assets.
##

# Conveniences to other module directories via relative paths
ARC_G1_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
"""Path to the model source directory."""
