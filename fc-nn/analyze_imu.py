#!/usr/bin/env python3
"""
analyze_imu.py — IMU noise characterisation: statistics, histograms, FFT/PSD.

Usage:
    python analyze_imu.py imu_log.csv
    python analyze_imu.py imu_log_static.csv imu_log_hover.csv   # overlay comparison

Outputs:
    • Console table   — mean, std, min/max, Shapiro-Wilk normality p-value
    • Figure 1        — Time series
    • Figure 2        — Histograms + Gaussian fit
    • Figure 3        — Welch PSD  (+ raw FFT magnitude)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch

# ─────────────────────────────────────────────────────────────
CHANNELS = [
    ("gx", "rad/s", "Gyro X"),
    ("gy", "rad/s", "Gyro Y"),
    ("gz", "rad/s", "Gyro Z"),
    ("ax", "g",     "Accel X"),
    ("ay", "g",     "Accel Y"),
    ("az", "g",     "Accel Z"),
]
NCHAN = len(CHANNELS)
COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c"]


# ─────────────────────────────────────────────────────────────
def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t, data) where data.shape = (N, 6)."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([
                float(row["t_s"]),
                float(row["gx_rads"]),
                float(row["gy_rads"]),
                float(row["gz_rads"]),
                float(row["ax_g"]),
                float(row["ay_g"]),
                float(row["az_g"]),
            ])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


def estimate_fs(t: np.ndarray) -> float:
    return 1.0 / float(np.median(np.diff(t)))


# ─────────────────────────────────────────────────────────────
def print_stats(label: str, t: np.ndarray, data: np.ndarray):
    fs = estimate_fs(t)
    N  = len(t)
    duration = t[-1] - t[0]
    print(f"\n{'═'*72}")
    print(f"  {label}  —  {N} samples  {duration:.1f} s  fs≈{fs:.1f} Hz")
    print(f"{'═'*72}")
    header = f"  {'Channel':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}  {'Normality p':>12}"
    print(header)
    print(f"  {'─'*68}")
    for i, (key, unit, name) in enumerate(CHANNELS):
        col = data[:, i]
        mu  = col.mean()
        sd  = col.std()
        lo  = col.min()
        hi  = col.max()
        # Shapiro-Wilk on up to 5000 samples (limit of the test)
        sample = col if len(col) <= 5000 else col[::max(1, len(col)//5000)]
        _, p = stats.shapiro(sample)
        flag = "" if p > 0.05 else "  ← NOT Gaussian"
        print(f"  {name:<10} {mu:>+12.5f} {sd:>12.5f} {lo:>+12.5f} {hi:>+12.5f} "
              f"  {p:>12.4f}{flag}  ({unit})")
    print()


# ─────────────────────────────────────────────────────────────
def fig_timeseries(datasets: list[tuple[str, np.ndarray, np.ndarray]]):
    fig, axes = plt.subplots(NCHAN, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("IMU Time Series", fontsize=14, fontweight="bold")

    linestyles = ["-", "--"]
    for i, (key, unit, name) in enumerate(CHANNELS):
        ax = axes[i]
        for j, (label, t, data) in enumerate(datasets):
            col = data[:, i]
            ax.plot(t, col, linestyles[j % 2], color=COLORS[i],
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
def fig_histograms(datasets: list[tuple[str, np.ndarray, np.ndarray]]):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("IMU Noise Histograms + Gaussian Fit", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    alphas = [0.65, 0.45]
    linestyles = ["-", "--"]

    for i, (key, unit, name) in enumerate(CHANNELS):
        ax = axes[i]
        for j, (label, t, data) in enumerate(datasets):
            col = data[:, i]
            mu, sd = col.mean(), col.std()
            # histogram
            n, bins, _ = ax.hist(col, bins=80, density=True,
                                 alpha=alphas[j], color=COLORS[i],
                                 label=f"{label}  σ={sd:.5f}")
            # Gaussian overlay
            x = np.linspace(bins[0], bins[-1], 300)
            ax.plot(x, stats.norm.pdf(x, mu, sd), linestyles[j],
                    color="black", linewidth=1.5)

        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel(unit, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
def fig_fft(datasets: list[tuple[str, np.ndarray, np.ndarray]]):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("IMU Noise — Welch Power Spectral Density", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    linestyles = ["-", "--"]

    for i, (key, unit, name) in enumerate(CHANNELS):
        ax = axes[i]
        for j, (label, t, data) in enumerate(datasets):
            fs = estimate_fs(t)
            col = data[:, i]
            # Welch PSD — 1/4 of signal per segment, 50% overlap
            nperseg = max(64, len(col) // 4)
            freqs, psd = welch(col, fs=fs, nperseg=nperseg, scaling="density")
            ax.semilogy(freqs, psd, linestyles[j], color=COLORS[i],
                        linewidth=1.2, label=label)

        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel(f"PSD ({unit}²/Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0, None)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze_imu.py <log.csv> [log2.csv ...]")

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

    fig1 = fig_timeseries(datasets)
    fig2 = fig_histograms(datasets)
    fig3 = fig_fft(datasets)

    # Save PNGs alongside the first CSV
    base = paths[0].parent / paths[0].stem
    fig1.savefig(f"{base}_timeseries.png", dpi=150)
    fig2.savefig(f"{base}_histograms.png", dpi=150)
    fig3.savefig(f"{base}_psd.png",        dpi=150)
    print(f"Saved: {base}_timeseries.png")
    print(f"Saved: {base}_histograms.png")
    print(f"Saved: {base}_psd.png")

    plt.show()


if __name__ == "__main__":
    main()
