# walk_noclock

unitree_rl_lab `unitree_g1_29dof_running`, run `2026-09-02_11-36-13_walk`
(`logs/rsl_rl/unitree_g1_29dof_running/2026-09-02_11-36-13_walk`, policy.onnx exported 2026-09-02 15:19).

Copied here so the deployment does not depend on a path inside the training logs.

| file | what it is |
|---|---|
| `policy.onnx` | the policy: input `obs` [1, 480], output `actions` [1, 29] |
| `deploy.yaml` | the exported deployment config every value in `configs/g1_sim2real.yaml` is derived from |
| `env.yaml` | the full IsaacLab env config of the training run |

## Differences from base_height / running_zerostop

| | walk_noclock | base_height, running_zerostop |
|---|---|---|
| observation | 96 per frame x 5 = **480** | 98 x 5 = 490 |
| `gait_phase` term | **absent** (`use_gait_phase: false`) | present, period 0.5 s, threshold 0.1 |
| elbow default pos | **0.97 rad**, both arms | 0.0 |
| vx command range | [-1.0, **2.0**] | [-1.0, 2.5] |
| command term | `velocity_commands` | `keyboard_velocity_commands` (zerostop) |

Everything else matches: `joint_ids_map`, `step_dt` 0.02, stiffness/damping, scales
(ang_vel 0.2, dof_pos 1.0, dof_vel 0.05, action 0.25), history length 5, term-major flattening.

The elbow offset matters: `joint_pos_rel` is measured against `default_joint_pos`, so
`arm_waist_target` in `configs/g1_sim2real.yaml` has to carry the same 0.97 or every arm
observation is offset by that much and the robot also holds the wrong pose before handover.

## Status

Not yet checked in sim2sim or on hardware. The config is pointed at it temporarily; the previous
policy is `../base_height/policy.onnx` (needs `use_gait_phase: true`, elbows 0.0, cmd_max vx 2.5).
