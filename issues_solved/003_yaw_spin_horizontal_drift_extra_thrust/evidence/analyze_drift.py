#!/usr/bin/env python3
"""
analyze_drift.py — evidence for issue 003.

Reads flight.csv (flight recorder, 100 Hz) next to this file and prints:
  1. lean, yaw rate, altitude and motor pair biases per 5 s window
  2. total heading change and how far a steady lean would carry the drone
  3. yaw response sign: does raising the CW pair (FR+RL) turn the drone CCW?
  4. hover thrust: mean motor command vs the kitchen-scale hover point

Run:  python analyze_drift.py > analysis.txt
"""
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
FS, G = 100, 9.81
HOVER_SCALE_US = 1372          # experiments/bench-2026-11-21/THRUST_RAMP.md, W = 1110 g
d = np.genfromtxt(HERE / "flight.csv", delimiter=",", names=True)
t = d["t_s"]
pw = {m: d["pwm_" + m] for m in ["FL", "FR", "RR", "RL"]}
lr = ((pw["FL"] + pw["RL"]) - (pw["FR"] + pw["RR"])) / 2     # left pair up  → +roll correction
fr = ((pw["FL"] + pw["FR"]) - (pw["RR"] + pw["RL"])) / 2     # front pair up
yw = ((pw["FR"] + pw["RL"]) - (pw["FL"] + pw["RR"])) / 2     # CW pair up    → CCW torque (training)
mean_m = np.mean(list(pw.values()), axis=0)

print("1. PER 5 s WINDOW (roll/pitch = AHRS deg; motor pair differences in µs)")
print("   window   roll  pitch   yaw°/s   alt m   L-R   F-R   CW-CCW   mean µs")
for a in range(0, 30, 5):
    m = (t >= a) & (t < a + 5)
    print(f"   {a:2d}-{a + 5:2d}s {d['roll'][m].mean():+6.1f} {d['pitch'][m].mean():+6.1f} "
          f"{np.degrees(d['gz'][m].mean()):+8.1f} {d['alt_m'][m].mean():7.2f} "
          f"{lr[m].mean():+5.0f} {fr[m].mean():+5.0f} {yw[m].mean():+8.0f} {mean_m[m].mean():9.0f}")
print(f"   whole    {d['roll'].mean():+6.1f} {d['pitch'].mean():+6.1f} "
      f"{np.degrees(d['gz'].mean()):+8.1f} {d['alt_m'].mean():7.2f} "
      f"{lr.mean():+5.0f} {fr.mean():+5.0f} {yw.mean():+8.0f} {mean_m.mean():9.0f}")

print("\n2. HEADING AND DRIFT")
print(f"   heading change over {t[-1]:.0f} s: {np.degrees(np.sum(d['gz']) / FS):+.0f}° (negative = clockwise)")
for lean in [1, 2, 3]:
    a = G * np.tan(np.radians(lean))
    print(f"   steady {lean}° lean, no position feedback: {a:.2f} m/s² → "
          f"{0.5 * a * 3**2:.1f} m after 3 s, {0.5 * a * 10**2:.1f} m after 10 s")

print("\n3. YAW RESPONSE SIGN (yaw angular accel vs CW-CCW, closed-loop so only the sign is meaningful)")
bl, al = signal.butter(2, 4, fs=FS)
acc = np.gradient(signal.filtfilt(bl, al, d["gz"])) * FS
for dl in [0, 4, 8]:
    y = np.roll(signal.filtfilt(bl, al, yw - yw.mean()), dl)
    m = slice(50, len(y) - 10)
    K = np.dot(y[m], acc[m]) / np.dot(y[m], y[m])
    print(f"   delay {dl * 10:3d} ms: {K:+.4f} rad/s² per µs   corr {np.corrcoef(y[m], acc[m])[0, 1]:+.2f}"
          f"   ({'CW pair up → CCW turn, matches training' if K > 0 else 'OPPOSITE of training'})")

print("\n4. HOVER THRUST")
u_s, u_f = (HOVER_SCALE_US - 1000) / 1000, (mean_m.mean() - 1000) / 1000
ratio = ((u_f + 0.015) / (u_s + 0.015)) ** 1.745
print(f"   mean motor {mean_m.mean():.0f} µs vs scale hover {HOVER_SCALE_US} µs → "
      f"{(ratio - 1) * 100:+.0f} % thrust on the measured curve (T ∝ (u+0.015)^1.745)")
print(f"   mean az {d['az'].mean():.3f} g (≈1 → not climbing), altitude {d['alt_m'].min():.2f}…{d['alt_m'].max():.2f} m (target 0.30)")
