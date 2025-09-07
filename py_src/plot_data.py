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

import numpy as np
import matplotlib.pyplot as plt

# Load the logged data
cur_pos_log = np.load("cur_pos_log.npy")  # Shape (time_steps, 29)
des_pos_log = np.load("time.npy")  # Shape (time_steps, 29)

# Check the shape of the data (should be time_steps x 29)
print(cur_pos_log.shape)
print(des_pos_log.shape)

# Generate the joint names
labels = [f'joint{i}' for i in range(1, 30)]

# Create a figure with a large size
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 15))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot the data
time_steps = np.arange(cur_pos_log.shape[0])

# Use a simple loop to plot each joint's current vs desired position
for i in range(1):
    ax = axes[i]
    ax.plot(des_pos_log[:], cur_pos_log[:, i], label=f'Current {labels[i]}', linestyle='-', color='b')
    # ax.plot(time_steps, des_pos_log[:], label=f'Desired {labels[i]}', linestyle='--', color='r')
    ax.set_title(labels[i])
    ax.set_xlabel("Time Steps",fontsize=10)
    ax.set_ylabel("Position",fontsize=10)
    ax.legend()

print(time_steps)

# Adjust layout to make sure there's no overlap
plt.tight_layout()
plt.show()
