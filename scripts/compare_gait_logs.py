#!/usr/bin/env python3
"""Compare wholebody_rl logs (sim or real) on the forward-lean signature.

usage: python3 scripts/compare_gait_logs.py LABEL=path/to/log.csv [LABEL=path ...] [--cmd 1.0] [--tol 0.05]

For each log, the RL-active rows whose cmd_x is within --tol of --cmd (after the
first 2 s at that command) are summarised:
  pitch      mean pelvis pitch [deg]            (sim ref -3, real 1.0 m/s +7)
  hip_amp    hip-pitch target amplitude [rad]   (sim ref 0.32, real 0.87)
  ank_qd_max max ankle-pitch target [rad]       (sim ref -0.35, real +0.30)
  ank_tau    peak |ankle pitch tau| [Nm]        (sim ref 13, real 32-41)
  hip_tau    peak |hip pitch tau| [Nm]          (sim ref 32, real 70)
  waist_tau  mean waist-pitch tau [Nm]          (sim ref -0.4, real -7)
Also prints pitch during the first 1.5 s after |cmd| first reaches 0.1 (the third-step tip).
"""
import sys, numpy as np, pandas as pd

args = [a for a in sys.argv[1:] if '=' in a and not a.startswith('--')]
cmd = float(sys.argv[sys.argv.index('--cmd') + 1]) if '--cmd' in sys.argv else 1.0
tol = float(sys.argv[sys.argv.index('--tol') + 1]) if '--tol' in sys.argv else 0.05
if not args:
    print(__doc__); sys.exit(1)

rows = []
for a in args:
    label, path = a.split('=', 1)
    d = pd.read_csv(path)
    rl = d[d.fsm_state == 3]
    if rl.empty:
        print(f'{label}: no RL-active rows'); continue
    at = rl[(rl.cmd_x - cmd).abs() <= tol]
    if len(at) > 5:
        at = at[at.time >= at.time.iloc[0] + 2.0]
    on = rl[rl.cmd_x.abs() >= 0.1]
    onset = ''
    if not on.empty:
        t0 = on.time.iloc[0]
        w = rl[(rl.time >= t0) & (rl.time <= t0 + 1.5)]
        onset = '%.1f -> %.1f (max %.1f)' % (np.degrees(w.pitch.iloc[0]), np.degrees(w.pitch.iloc[-1]), np.degrees(w.pitch.max()))
    if len(at) < 5:
        rows.append(dict(log=label, n=len(at), onset_pitch=onset)); continue
    hip_amp = np.mean([at.qdes0.max() - at.qdes0.min(), at.qdes6.max() - at.qdes6.min()])
    rows.append(dict(log=label, n=len(at), secs=round(at.time.iloc[-1] - at.time.iloc[0], 1),
                     pitch=round(np.degrees(at.pitch.mean()), 1), pitch_max=round(np.degrees(at.pitch.max()), 1),
                     hip_amp=round(hip_amp, 2), ank_qd_max=round(max(at.qdes4.max(), at.qdes10.max()), 2),
                     ank_tau=round(max(at.tau4.abs().max(), at.tau10.abs().max()), 1),
                     hip_tau=round(max(at.tau0.abs().max(), at.tau6.abs().max()), 1),
                     knee_tau=round(max(at.tau3.abs().max(), at.tau9.abs().max()), 1),
                     waist_tau=round(at.tau14.mean(), 1), onset_pitch=onset))
pd.set_option('display.width', 250)
print(f'cmd_x = {cmd} +- {tol}')
print(pd.DataFrame(rows).to_string(index=False))
