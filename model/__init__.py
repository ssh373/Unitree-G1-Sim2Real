"""******************************************************************************************
* HumARConoid-Sim2Real
*
* Sim2Real Transfer to Optimize Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid-Sim2Real.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
******************************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

"""Package containing asset and sensor configurations."""

import os
import toml

##
# Configuration for different assets.
##

# Conveniences to other module directories via relative paths
ARC_G1_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
"""Path to the model source directory."""

