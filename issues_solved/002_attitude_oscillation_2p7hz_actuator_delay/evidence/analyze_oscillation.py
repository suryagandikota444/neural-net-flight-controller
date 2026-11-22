#!/usr/bin/env python3
"""
analyze_oscillation.py — evidence for issue 002.

Reads flight.csv (flight recorder, 100 Hz) next to this file and prints:
  1. spectrum peaks of body rates / attitude / accel (is there a resonance?)
  2. how the 2-5 Hz oscillation waxes and wanes (2 s windows)
  3. motor pair biases (roll / pitch / yaw)
  4. actuator-delay fit: angular acceleration vs motor command, for a grid
     of pure delays and first-order motor lags. R² tells which delay
     best explains the motion.

Run:  python analyze_oscillation.py > analysis.txt
"""
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
FS = 100
d = np.genfromtxt(HERE / "flight.csv", delimiter=",", names=True)
pw = {m: d["pwm_" + m] for m in ["FL", "FR", "RR", "RL"]}

print("1. SPECTRUM PEAKS (Welch, 5.12 s windows) — relative height in ()")
for k in ["gx", "gy", "gz", "roll", "pitch", "ax", "ay", "az"]:
    f, p = signal.welch(d[k] - d[k].mean(), fs=FS, nperseg=512)
    band = (f > 0.3) & (f < 50)
    idx = signal.find_peaks(p[band], prominence=p[band].max() * 0.08)[0]
    top = sorted(idx, key=lambda i: -p[band][i])[:3]
    rms = np.sqrt(np.trapezoid(p[band], f[band]))
    print(f"   {k:5s} rms {rms:6.3f}   " +
          ", ".join(f"{f[band][i]:.2f} Hz ({p[band][i] / p[band].max():.2f})" for i in top))
for k in ["gx", "gy"]:
    f, p = signal.welch(d[k] - d[k].mean(), fs=FS, nperseg=512)
    b, a = (f > 2) & (f < 5), (f > 0.3) & (f < 50)
    print(f"   {k}: {np.trapezoid(p[b], f[b]) / np.trapezoid(p[a], f[a]) * 100:.0f} % of its power is in 2-5 Hz")

print("\n2. 2-5 Hz OSCILLATION SIZE PER 2 s WINDOW (rad/s envelope)")
bb, ab = signal.butter(2, [2, 5], btype="band", fs=FS)
for k in ["gx", "gy"]:
    env = np.abs(signal.hilbert(signal.filtfilt(bb, ab, d[k])))
    print(f"   {k}: " + " ".join(f"{env[i:i + 200].mean():.2f}" for i in range(0, len(env), 200)))

print("\n3. MOTOR BIASES (mean µs)")
print("   " + "  ".join(f"{m} {v.mean():.0f}" for m, v in pw.items()))
print(f"   left-right {((pw['FL'] + pw['RL']) - (pw['FR'] + pw['RR'])).mean() / 2:+.1f}"
      f"   front-rear {((pw['FL'] + pw['FR']) - (pw['RR'] + pw['RL'])).mean() / 2:+.1f}"
      f"   CW-CCW (FR+RL vs FL+RR) {((pw['FR'] + pw['RL']) - (pw['FL'] + pw['RR'])).mean() / 2:+.1f}")
print(f"   mean roll {d['roll'].mean():+.2f}°, pitch {d['pitch'].mean():+.2f}°, yaw rate {d['gz'].mean():+.3f} rad/s")

print("\n4. ACTUATOR DELAY FIT  (angular accel = K · delayed/lagged motor command)")
u_roll = ((pw["FL"] + pw["RL"]) - (pw["FR"] + pw["RR"])) / 2     # left up → +roll
u_pitch = ((pw["RR"] + pw["RL"]) - (pw["FL"] + pw["FR"])) / 2    # rear up → +pitch
bl, al = signal.butter(2, 8, fs=FS)


def lag1(u, tau):
    if tau == 0:
        return u.copy()
    k = 1 - np.exp(-1 / (FS * tau))
    y, s = np.zeros_like(u), u[0]
    for i, v in enumerate(u):
        s += k * (v - s)
        y[i] = s
    return y


for name, u, rate in [("roll", u_roll, d["gx"]), ("pitch", u_pitch, d["gy"])]:
    acc = np.gradient(signal.filtfilt(bl, al, rate)) * FS
    fits = []
    for tau in [0, .01, .02, .03, .04, .05, .06, .08, .1]:
        for dl in range(6):
            y = signal.filtfilt(bl, al, lag1(u - u.mean(), tau))
            y = np.roll(y, dl); y[:dl] = 0
            m = slice(50, len(acc) - 10)
            K = np.dot(y[m], acc[m]) / np.dot(y[m], y[m])
            r2 = 1 - (acc[m] - K * y[m]).var() / acc[m].var()
            fits.append((r2, tau, dl, K))
    fits.sort(reverse=True)
    inst = [f for f in fits if f[1] == 0 and f[2] == 0][0]
    print(f"   {name}: instant response (sim assumption)  R² {inst[0]:.2f}")
    for r2, tau, dl, K in fits[:3]:
        print(f"   {name}: dead time {dl * 10:2d} ms + lag τ {tau * 1000:3.0f} ms  R² {r2:.2f}  "
              f"gain {K:.3f} rad/s² per µs")
