# running_zerostop

unitree_rl_lab `unitree_g1_29dof_running`, run `2026-09-11_18-14-36_zerostop`.

Copied here so the deployment does not depend on a path inside the training logs.

| file | what it is |
|---|---|
| `policy.onnx` | the policy: input `obs` [1, 490], output `actions` [1, 29], opset 18 |
| `deploy.yaml` | the exported deployment config every value in `configs/g1_sim2real.yaml` is derived from |
| `env.yaml` | the full IsaacLab env config of the training run |

## Shape of the policy

- 29-DoF whole-body joint position targets (legs, waist and arms are all driven by the policy)
- Observation: 98 per frame x 5 frames of history = 490, laid out term-major, oldest to newest
  `[ang_vel x5][gravity x5][cmd x5][joint_pos_rel x5][joint_vel_rel x5][last_action x5][gait_phase x5]`
- Policy rate 50 Hz (`step_dt` 0.02), gait period 0.5 s, clock masked to (0,0) below |cmd| = 0.1
- Command range seen in training: vx [-1.0, 2.5], vy [-1.0, 1.0], wz [-1.0, 1.0]

Requires ONNX Runtime >= 1.14 (opset 18, `GetInputNameAllocated`).

## Sim2sim check (unitree_mujoco, 2026-09-15)

68 s at vx 0.5 m/s: stride held at 2.02 Hz, peak knee torque 126.6 Nm against a 134.8 Nm clamp
and a 139 Nm kill, no clamp or kill event on any joint, upper-body speed peaked at 8.3 of
15 rad/s.

Known behaviour, not a deployment fault: with a zero velocity command the policy settles into a
crouch (knee ~30 deg against a 17 deg default pose) and keeps the knees deep while walking
(~67 deg). Nothing in the training reward pins down the standing posture -- `base_height_l2`
carries a weight of only -2.0 and there is no deviation penalty on knee or hip pitch. Being
retrained with a larger height weight.
