#!/usr/bin/env python3
"""
log_imu.py — Capture full telemetry lines from firmware and write to CSV.

Firmware must be built with RAW_LOG_MODE=1 and the TEL print block enabled.

Format from firmware:
  TEL,gx,gy,gz,ax,ay,az,roll,pitch,obs0..obs11,nn0..nn3,mot0..mot3

Usage:
    python log_imu.py --port /dev/cu.SLAB_USBtoUART --seconds 30
    python log_imu.py --port /dev/cu.SLAB_USBtoUART --seconds 30 --out flight_test.csv

Requires:  pip install pyserial
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    sys.exit("pyserial not found — run:  pip install pyserial")


COLUMNS = [
    "t_s",
    # Raw IMU (6)
    "gx_rads", "gy_rads", "gz_rads",
    "ax_g", "ay_g", "az_g",
    # AHRS attitude (2)
    "roll_deg", "pitch_deg",
    # Observation vector (12)
    "obs_0", "obs_1", "obs_2", "obs_3",
    "obs_4", "obs_5", "obs_6", "obs_7",
    "obs_8", "obs_9", "obs_10", "obs_11",
    # NN output post-smoothing (4)
    "nn_0", "nn_1", "nn_2", "nn_3",
    # Final motor commands (4)
    "mot_0", "mot_1", "mot_2", "mot_3",
]

N_VALUES = len(COLUMNS) - 1  # 28 float values expected after "TEL,"


def parse_args():
    p = argparse.ArgumentParser(description="Log full flight telemetry from FC firmware")
    p.add_argument("--port", default="/dev/cu.SLAB_USBtoUART",
                   help="Serial port (default: /dev/cu.SLAB_USBtoUART)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=30.0,
                   help="Duration to log in seconds (default: 30)")
    p.add_argument("--out", default=None,
                   help="Output CSV filename (default: flight_log_YYYYMMDD_HHMMSS.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"flight_log_{stamp}.csv"

    out_path = Path(args.out)
    print(f"Port     : {args.port} @ {args.baud}")
    print(f"Duration : {args.seconds} s")
    print(f"Output   : {out_path}")
    print(f"Expected : {N_VALUES} values per line (TEL,...)")
    print()
    print("Waiting for TEL lines — firmware must have the TEL print block enabled")
    print("Press Ctrl-C to stop early.")
    print()

    rows = 0
    bad_lines = 0
    t_start = None

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        sys.exit(f"Cannot open {args.port}: {e}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        try:
            while True:
                line = ser.readline().decode("ascii", errors="replace").strip()
                if not line.startswith("TEL,"):
                    # Also print non-TEL lines so you can see other firmware output
                    if line:
                        print(f"  [{line[:80]}]")
                    continue

                parts = line.split(",")
                # TEL + 28 values = 29 parts
                if len(parts) != N_VALUES + 1:
                    bad_lines += 1
                    if bad_lines <= 5:
                        print(f"  WARN: expected {N_VALUES+1} fields, got {len(parts)}: {line[:60]}...")
                    continue

                try:
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    bad_lines += 1
                    continue

                now = time.monotonic()
                if t_start is None:
                    t_start = now
                    print(f"  First TEL sample received — logging for {args.seconds} s ...",
                          flush=True)

                elapsed = now - t_start
                if elapsed > args.seconds:
                    break

                writer.writerow([f"{elapsed:.4f}"] + [f"{v:.4f}" for v in vals])
                rows += 1

                if rows % 50 == 0:
                    pct = elapsed / args.seconds * 100
                    # Show a quick summary of what we're seeing
                    roll, pitch = vals[6], vals[7]
                    mots = vals[24:28]
                    print(f"  {rows:5d} samples  {elapsed:5.1f}s ({pct:3.0f}%)  "
                          f"R:{roll:+6.1f} P:{pitch:+6.1f}  "
                          f"MOT:[{mots[0]:.2f} {mots[1]:.2f} {mots[2]:.2f} {mots[3]:.2f}]",
                          flush=True)

        except KeyboardInterrupt:
            print("\nStopped early by user.")

    ser.close()
    duration = time.monotonic() - t_start if t_start else 0
    print(f"\nDone. {rows} samples saved to {out_path}")
    if bad_lines:
        print(f"Dropped {bad_lines} malformed lines.")
    if rows == 0:
        print("WARNING: No TEL samples captured. Is the firmware TEL block enabled?")
    else:
        actual_hz = rows / duration
        print(f"Effective sample rate: {actual_hz:.1f} Hz")


if __name__ == "__main__":
    main()