#!/usr/bin/env python3
"""
analyze_imu.py — Full flight telemetry analysis.

Reads CSVs produced by log_imu.py (TEL format with 28 channels):
  Raw IMU (6), AHRS attitude (2), obs vector (12), NN output (4), motor cmd (4)

Usage:
    python analyze_imu.py flight_log.csv
    python analyze_imu.py flight_a.csv flight_b.csv   # overlay comparison

Outputs:
    • Console stats table
    • Figure 1: Raw IMU time series (gyro + accel)
    • Figure 2: AHRS attitude (roll/pitch) time series
    • Figure 3: Observation vector time series (12 channels)
    • Figure 4: NN outputs vs final motor commands
    • Figure 5: Motor command histograms
    • Figure 6: Gyro PSD (frequency analysis)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch


# ─────────────────────────────────────────────────────────────
# Column definitions matching log_imu.py COLUMNS
# ─────────────────────────────────────────────────────────────
ALL_COLUMNS = [
    "t_s",
    "gx_rads", "gy_rads", "gz_rads",
    "ax_g", "ay_g", "az_g",
    "roll_deg", "pitch_deg",
    "obs_0", "obs_1", "obs_2", "obs_3",
    "obs_4", "obs_5", "obs_6", "obs_7",
    "obs_8", "obs_9", "obs_10", "obs_11",
    "nn_0", "nn_1", "nn_2", "nn_3",
    "mot_0", "mot_1", "mot_2", "mot_3",
]

# Semantic groups for plotting
IMU_CHANNELS = [
    (1, "gx", "rad/s", "Gyro X", "#e74c3c"),
    (2, "gy", "rad/s", "Gyro Y", "#2ecc71"),
    (3, "gz", "rad/s", "Gyro Z", "#3498db"),
    (4, "ax", "g",     "Accel X", "#e67e22"),
    (5, "ay", "g",     "Accel Y", "#9b59b6"),
    (6, "az", "g",     "Accel Z", "#1abc9c"),
]

AHRS_CHANNELS = [
    (7, "roll",  "deg", "Roll",  "#e74c3c"),
    (8, "pitch", "deg", "Pitch", "#3498db"),
]

OBS_LABELS = [
    "gx_norm", "gy_norm", "gz_norm",
    "ax_norm", "ay_norm", "az_norm",
    "roll_err", "pitch_err",
    "alt_err", "vvel",
    "vx (0)", "vy (0)",
]

NN_LABELS  = ["NN[0] FL", "NN[1] FR", "NN[2] RR", "NN[3] RL"]
MOT_LABELS = ["Mot[0] FL", "Mot[1] FR", "Mot[2] RR", "Mot[3] RL"]

OBS_COLORS = plt.cm.tab20(np.linspace(0, 1, 12))
NN_COLORS  = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
MOT_COLORS = ["#c0392b", "#27ae60", "#2980b9", "#d35400"]


# ─────────────────────────────────────────────────────────────
def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t, data) where data.shape = (N, 28)."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(row[c]) for c in ALL_COLUMNS]
            rows.append(vals)
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]  # t, data (28 cols)


def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    return 1.0 / float(np.median(dt[dt > 0]))


# ─────────────────────────────────────────────────────────────
def print_stats(label: str, t: np.ndarray, data: np.ndarray):
    fs = estimate_fs(t)
    N  = len(t)
    duration = t[-1] - t[0]
    print(f"\n{'═'*78}")
    print(f"  {label}  —  {N} samples  {duration:.1f} s  fs≈{fs:.1f} Hz")
    print(f"{'═'*78}")

    groups = [
        ("Raw IMU", [(0, "Gyro X", "rad/s"), (1, "Gyro Y", "rad/s"), (2, "Gyro Z", "rad/s"),
                     (3, "Accel X", "g"), (4, "Accel Y", "g"), (5, "Accel Z", "g")]),
        ("AHRS",    [(6, "Roll", "deg"), (7, "Pitch", "deg")]),
        ("NN Out",  [(20, "NN[0]", ""), (21, "NN[1]", ""), (22, "NN[2]", ""), (23, "NN[3]", "")]),
        ("Motors",  [(24, "Mot[0]", ""), (25, "Mot[1]", ""), (26, "Mot[2]", ""), (27, "Mot[3]", "")]),
    ]

    for group_name, channels in groups:
        print(f"\n  ── {group_name} ──")
        header = f"  {'Channel':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}"
        print(header)
        print(f"  {'─'*64}")
        for idx, name, unit in channels:
            col = data[:, idx]
            mu, sd = col.mean(), col.std()
            lo, hi = col.min(), col.max()
            u = f" {unit}" if unit else ""
            print(f"  {name:<12} {mu:>+10.4f} {sd:>10.4f} {lo:>+10.4f} {hi:>+10.4f} {hi-lo:>10.4f}{u}")
    print()


# ─────────────────────────────────────────────────────────────
# FIGURE 1: Raw IMU time series
# ─────────────────────────────────────────────────────────────
def fig_raw_imu(datasets):
    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Raw IMU Time Series", fontsize=14, fontweight="bold")

    for i, (col_idx, key, unit, name, color) in enumerate(IMU_CHANNELS):
        ax = axes[i]
        for j, (label, t, data) in enumerate(datasets):
            ls = "-" if j == 0 else "--"
            ax.plot(t, data[:, col_idx-1], ls, color=color,
                    alpha=0.7, linewidth=0.6, label=label)
        ax.set_ylabel(f"{name}\n({unit})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 2: AHRS Attitude (roll/pitch)
# ─────────────────────────────────────────────────────────────
def fig_attitude(datasets):
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle("AHRS Attitude — Roll & Pitch", fontsize=14, fontweight="bold")

    for i, (col_idx, key, unit, name, color) in enumerate(AHRS_CHANNELS):
        ax = axes[i]
        for j, (label, t, data) in enumerate(datasets):
            ls = "-" if j == 0 else "--"
            ax.plot(t, data[:, col_idx-1], ls, color=color,
                    alpha=0.8, linewidth=0.8, label=label)
        ax.set_ylabel(f"{name} ({unit})", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 3: Observation vector (12 channels, grouped)
# ─────────────────────────────────────────────────────────────
def fig_obs_vector(datasets):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Observation Vector (NN Inputs)", fontsize=14, fontweight="bold")

    # Group: gyro obs (0-2), accel obs (3-5), attitude obs (6-7), alt/vel obs (8-11)
    groups = [
        ("Gyro (normalized)", [0, 1, 2]),
        ("Accel (normalized)", [3, 4, 5]),
        ("Attitude error (roll/pitch)", [6, 7]),
        ("Altitude / Velocity / Horiz", [8, 9, 10, 11]),
    ]

    for ax_i, (title, indices) in enumerate(groups):
        ax = axes[ax_i]
        for idx in indices:
            col_in_data = 8 + idx  # obs starts at column index 8 in data array
            for j, (label, t, data) in enumerate(datasets):
                ls = "-" if j == 0 else "--"
                lbl = OBS_LABELS[idx] if j == 0 else f"{label}: {OBS_LABELS[idx]}"
                ax.plot(t, data[:, col_in_data], ls,
                        color=OBS_COLORS[idx], alpha=0.8, linewidth=0.7, label=lbl)
        ax.set_title(title, fontsize=9, loc="left")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.axhline(1, color="red", linewidth=0.3, alpha=0.3, linestyle=":")
        ax.axhline(-1, color="red", linewidth=0.3, alpha=0.3, linestyle=":")
        ax.set_ylabel("Normalized", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 4: NN output vs motor commands
# ─────────────────────────────────────────────────────────────
def fig_nn_vs_motors(datasets):
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("NN Outputs (smoothed) vs Final Motor Commands", fontsize=14, fontweight="bold")

    for i in range(4):
        ax = axes[i]
        nn_col = 20 + i   # nn_0..nn_3
        mot_col = 24 + i  # mot_0..mot_3

        for j, (label, t, data) in enumerate(datasets):
            pfx = f"{label}: " if len(datasets) > 1 else ""
            ax.plot(t, data[:, nn_col], "-", color=NN_COLORS[i],
                    alpha=0.7, linewidth=0.7, label=f"{pfx}{NN_LABELS[i]}")
            ax.plot(t, data[:, mot_col], "-", color=MOT_COLORS[i],
                    alpha=0.9, linewidth=1.0, label=f"{pfx}{MOT_LABELS[i]}")

        ax.set_ylabel(NN_LABELS[i].split()[0], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 5: Motor command distributions
# ─────────────────────────────────────────────────────────────
def fig_motor_histograms(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Motor Command Distributions", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        mot_col = 24 + i
        for j, (label, t, data) in enumerate(datasets):
            col = data[:, mot_col]
            mu, sd = col.mean(), col.std()
            alpha = 0.65 if j == 0 else 0.45
            ax.hist(col, bins=60, density=True, alpha=alpha,
                    color=MOT_COLORS[i], label=f"{label}  μ={mu:.3f} σ={sd:.3f}")
        ax.set_title(MOT_LABELS[i], fontsize=10)
        ax.set_xlabel("Command (0-1)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 6: Gyro PSD (vibration fingerprint)
# ─────────────────────────────────────────────────────────────
def fig_gyro_psd(datasets):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Gyro Power Spectral Density (vibration fingerprint)", fontsize=14, fontweight="bold")

    gyro_info = [
        (0, "Gyro X", "#e74c3c"),
        (1, "Gyro Y", "#2ecc71"),
        (2, "Gyro Z", "#3498db"),
    ]

    for i, (col_idx, name, color) in enumerate(gyro_info):
        ax = axes[i]
        for j, (label, t, data) in enumerate(datasets):
            fs = estimate_fs(t)
            col = data[:, col_idx]
            nperseg = max(64, len(col) // 4)
            freqs, psd = welch(col, fs=fs, nperseg=nperseg, scaling="density")
            ls = "-" if j == 0 else "--"
            ax.semilogy(freqs, psd, ls, color=color, linewidth=1.2, label=label)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD (rad/s)²/Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze_imu.py <flight_log.csv> [log2.csv ...]")

    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    datasets = []
    for p in paths:
        print(f"Loading {p} ...")
        t, data = load_csv(p)
        label = p.stem
        datasets.append((label, t, data))
        print_stats(label, t, data)

    print("Generating figures ...")

    figs = [
        ("raw_imu",      fig_raw_imu(datasets)),
        ("attitude",     fig_attitude(datasets)),
        ("obs_vector",   fig_obs_vector(datasets)),
        ("nn_vs_motors", fig_nn_vs_motors(datasets)),
        ("motor_hist",   fig_motor_histograms(datasets)),
        ("gyro_psd",     fig_gyro_psd(datasets)),
    ]

    base = paths[0].parent / paths[0].stem
    for suffix, fig in figs:
        fname = f"{base}_{suffix}.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved: {fname}")

    plt.show()


if __name__ == "__main__":
    main()