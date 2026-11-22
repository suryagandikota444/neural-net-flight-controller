#!/usr/bin/env python3
"""
pull_log.py — pull the flight recorder log off the drone over USB (DFU).

    1. Hold BOOT, plug in USB (DFU mode — same as flashing).
    2. make pull-log          (or: python fc-nn/pull_log.py)
    3. Unplug / replug USB to run the drone normally.

What it does:
  - reads flash sector 11 (where blackbox.c saved the last 30 s of flight)
    with `dfu-util -U`,
  - checks it (magic, size, checksum),
  - writes experiments/blackbox/<YYYY-MM-DD_HHMMSS>/
        flight.csv      one row per 10 ms tick, physical units
        flight.png      attitude / rates / motors / NN output / altitude
        summary.txt     when it tipped, wobble frequency, growing or not, ...
        flight_raw.bin  the raw sector (for re-decoding later)
  - then ERASES the log from the drone (only after everything above
    succeeded). Use --keep to leave it on the drone.

Re-decode a saved raw file:  python fc-nn/pull_log.py --bin experiments/blackbox/<dir>/flight_raw.bin
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

LOG_ADDR = 0x080E0000          # flash sector 11 (blackbox.c BB_FLASH_ADDR)
LOG_SIZE = 128 * 1024
MAGIC = 0x31584242             # "BBX1"
VERSION = 1
HDR_WORDS = 8
REC_FMT = "<H3h3h2h4h4H2hH"    # must match bb_record_t in blackbox.h
REC_SIZE = struct.calcsize(REC_FMT)
MODES = {0: "HOVER", 1: "STICK", 2: "LAND"}
MOTORS = ["FL", "FR", "RR", "RL"]
COLUMNS = (["t_s", "tick", "gx", "gy", "gz", "ax", "ay", "az", "roll", "pitch"]
           + [f"nn_{m}" for m in MOTORS] + [f"pwm_{m}" for m in MOTORS]
           + ["alt_m", "vvel_ms", "ramping", "safety_cut", "link_ok", "mode",
              "baro_ok", "level_cal", "loop_us"])


# ── DFU ────────────────────────────────────────────────────────────────
def dfu(*args: str) -> subprocess.CompletedProcess:
    exe = shutil.which("dfu-util")
    if exe is None:
        sys.exit("dfu-util not found (brew install dfu-util)")
    return subprocess.run([exe, "-a", "0", *args], capture_output=True, text=True)


def read_sector(dest: Path, length: int = LOG_SIZE) -> bytes:
    r = dfu("-s", f"0x{LOG_ADDR:08X}:{length}", "-U", str(dest))
    if r.returncode != 0 or not dest.exists():
        msg = (r.stderr or r.stdout).strip().splitlines()
        sys.exit("Could not read the drone over USB.\n"
                 "  Is it in DFU mode? Hold BOOT while plugging in USB (like flashing).\n"
                 "  dfu-util said: " + (msg[-1] if msg else "?"))
    return dest.read_bytes()


def erase_log(tmp: Path) -> bool:
    """Writing 4 bytes of 0xFF makes dfu-util erase the whole sector first."""
    blank = tmp / "erase.bin"
    blank.write_bytes(b"\xff" * 4)
    r = dfu("-s", f"0x{LOG_ADDR:08X}", "-D", str(blank))
    if r.returncode != 0:
        return False
    check = read_sector(tmp / "verify.bin", 32)
    return struct.unpack("<I", check[:4])[0] != MAGIC


# ── decode ─────────────────────────────────────────────────────────────
class NoLog(Exception):
    pass


def parse(raw: bytes) -> tuple[dict, np.ndarray]:
    hdr = struct.unpack(f"<{HDR_WORDS}I", raw[:HDR_WORDS * 4])
    if hdr[0] != MAGIC:
        raise NoLog()
    magic, ver, size, count, csum, flight_ticks = hdr[:6]
    if ver != VERSION or size != REC_SIZE or not 1 <= count <= 3000:
        raise ValueError(f"unexpected header: version {ver}, record size {size}, count {count}")
    blob = raw[HDR_WORDS * 4: HDR_WORDS * 4 + count * REC_SIZE]
    if (sum(struct.unpack(f"<{len(blob)//4}I", blob)) & 0xFFFFFFFF) != csum:
        raise ValueError("checksum mismatch — log is corrupted")

    rows, t, prev = [], 0.0, None
    for i in range(count):
        f = struct.unpack_from(REC_FMT, blob, i * REC_SIZE)
        tick = f[0]
        if prev is not None:
            t += ((tick - prev) & 0xFFFF) * 0.01
        prev = tick
        fl = f[19]
        rows.append([t, tick,
                     *(np.array(f[1:4]) / 1000.0), *(np.array(f[4:7]) / 1000.0),
                     f[7] / 100.0, f[8] / 100.0,
                     *(np.array(f[9:13]) / 10000.0), *f[13:17],
                     f[17] / 1000.0, f[18] / 1000.0,
                     fl & 1, (fl >> 1) & 1, (fl >> 2) & 1, (fl >> 3) & 3, (fl >> 5) & 1, (fl >> 6) & 1,
                     ((fl >> 8) & 0xFF) * 50])
    return dict(count=count, flight_ticks=flight_ticks), np.array(rows, dtype=float)


# ── analysis ───────────────────────────────────────────────────────────
def dominant_freq(x: np.ndarray, dt: float = 0.01) -> float:
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), dt)
    band = (freqs > 0.5) & (freqs < 20)
    return float(freqs[band][np.argmax(spec[band])])


def summarize(d: np.ndarray, info: dict) -> list[str]:
    c = {name: d[:, i] for i, name in enumerate(COLUMNS)}
    t, n = c["t_s"], len(d)
    out = ["FLIGHT RECORDER SUMMARY", "=" * 40,
           f"recorded: last {t[-1]:.1f} s of {info['flight_ticks'] / 100:.1f} s armed"]
    gaps = np.diff(c["tick"]) % 65536
    if np.any(gaps != 1):
        out.append(f"loop stalls: {int(np.sum(gaps != 1))}, longest {gaps.max() * 10:.0f} ms")
    out.append(f"loop time: median {np.median(c['loop_us']):.0f} µs, max {c['loop_us'].max():.0f} µs")
    modes = sorted({MODES.get(int(m), "?") for m in c["mode"]})
    out.append(f"mode: {', '.join(modes)}   radio link lost: {int(np.sum(c['link_ok'] == 0))} ticks")
    out.append("level reference: " + ("stored calibration (TEST 18)" if c["level_cal"].all()
                                      else "boot surface (no stored calibration)"))

    cut = np.nonzero(c["safety_cut"])[0]
    end = cut[0] if len(cut) else n
    out.append(f"end: SAFETY CUT at t={t[end]:.2f} s (roll {c['roll'][end]:+.0f}°, pitch {c['pitch'][end]:+.0f}°)"
               if len(cut) else "end: disarmed with the ARM switch / failsafe")

    att = np.maximum(np.abs(c["roll"]), np.abs(c["pitch"]))
    for lim in (5, 10, 20, 45):
        idx = np.nonzero(att[:end] > lim)[0]
        if len(idx):
            k = idx[0]
            out.append(f"first tilt > {lim:>2}°: t={t[k]:6.2f} s (roll {c['roll'][k]:+6.1f}, pitch {c['pitch'][k]:+6.1f})")

    seg = np.arange(max(0, end - 400), end)
    seg = seg[c["ramping"][seg] == 0]
    if len(seg) > 100:
        out += ["", "last ~4 s before the end:"]
        for name, unit in (("roll", "°"), ("pitch", "°"), ("gx", "rad/s"), ("gy", "rad/s"), ("gz", "rad/s")):
            x = c[name][seg]
            h = len(x) // 2
            r1, r2 = float(np.std(x[:h])), float(np.std(x[h:]))
            trend = "GROWING" if r2 > 1.5 * r1 else ("decaying" if r2 < 0.67 * r1 else "steady")
            fs = f"{dominant_freq(x):5.2f} Hz" if max(r1, r2) > 1e-3 else "    —   "
            out.append(f"  {name:>5}: {fs}  size {r1:6.2f} → {r2:6.2f} {unit}  ({trend})")
        nn = np.stack([c[f"nn_{m}"][seg] for m in MOTORS], 1)
        pw = np.stack([c[f"pwm_{m}"][seg] for m in MOTORS], 1)
        out.append(f"  NN output at its ±1 limit: {np.mean(np.abs(nn) > 0.99) * 100:.1f} % of samples")
        out.append(f"  motors: {pw.min():.0f}–{pw.max():.0f} µs, mean {pw.mean():.0f} (hover ≈ 1372)")
        out.append(f"  altitude: {c['alt_m'][seg].min():+.2f} … {c['alt_m'][seg].max():+.2f} m")
        out.append(f"  accel z std: {np.std(c['az'][seg]):.3f} g (training noise 0.22 g)")
    return out


def plot(d: np.ndarray, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = {name: d[:, i] for i, name in enumerate(COLUMNS)}
    t = c["t_s"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    ax[0].plot(t, c["roll"], label="roll"); ax[0].plot(t, c["pitch"], label="pitch")
    ax[0].set(ylabel="deg", title="Attitude (AHRS)")
    for k in ("gx", "gy", "gz"):
        ax[1].plot(t, c[k], label=k)
    ax[1].set(ylabel="rad/s", title="Body rates")
    for m in MOTORS:
        ax[2].plot(t, c[f"pwm_{m}"], label=m)
    ax[2].set(ylabel="µs", title="Motor commands")
    for m in MOTORS:
        ax[3].plot(t, c[f"nn_{m}"], label=m)
    ax[3].set(ylabel="NN out", title="Raw NN output (before filter)")
    ax[4].plot(t, c["alt_m"], label="alt (m)"); ax[4].plot(t, c["vvel_ms"], label="vvel (m/s)")
    ax[4].set(xlabel="time (s)", title="Barometer")
    cut = np.nonzero(c["safety_cut"])[0]
    for a in ax:
        a.grid(alpha=0.3); a.legend(loc="upper left")
        if len(cut):
            a.axvline(t[cut[0]], color="r", ls="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def save_outputs(raw: bytes, info: dict, d: np.ndarray, out_root: Path) -> Path:
    folder = out_root / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "flight_raw.bin").write_bytes(raw)
    np.savetxt(folder / "flight.csv", d, delimiter=",", header=",".join(COLUMNS), comments="",
               fmt=["%.2f", "%d"] + ["%.4f"] * (len(COLUMNS) - 2))
    lines = summarize(d, info)
    (folder / "summary.txt").write_text("\n".join(lines) + "\n")
    plot(d, folder / "flight.png")
    print("\n".join(lines))
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep", action="store_true", help="do not erase the log from the drone")
    p.add_argument("--bin", type=Path, help="decode a saved flight_raw.bin instead of reading the drone")
    p.add_argument("--out", type=Path, default=REPO / "experiments" / "blackbox")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = args.bin.read_bytes() if args.bin else read_sector(tmp / "flight_raw.bin")
        try:
            info, d = parse(raw)
        except NoLog:
            print("No flight log stored on the drone (nothing to pull).")
            if not args.bin:
                print("Unplug / replug USB to run the drone normally.")
            return
        except ValueError as e:
            if not args.bin:
                keep = args.out / ("corrupt_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".bin")
                keep.parent.mkdir(parents=True, exist_ok=True)
                keep.write_bytes(raw)
                print(f"raw sector kept in {keep}; the log was NOT erased from the drone")
            sys.exit(f"Log unreadable: {e}")

        folder = save_outputs(raw, info, d, args.out)
        print(f"\nsaved → {folder.relative_to(REPO) if folder.is_relative_to(REPO) else folder}/")
        print("        flight.csv  flight.png  summary.txt  flight_raw.bin")

        if args.bin:
            return
        if args.keep:
            print("--keep: log left on the drone.")
        elif erase_log(tmp):
            print("Log erased from the drone.")
        else:
            print("WARNING: could not erase the log from the drone (it is saved on the Mac).")
        print("Unplug / replug USB to run the drone normally.")


if __name__ == "__main__":
    main()
