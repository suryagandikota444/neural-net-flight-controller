#!/usr/bin/env python3
"""
log_imu.py — Capture RAW IMU lines from firmware and write to CSV.

Firmware must be built with RAW_LOG_MODE=1:
    make CFLAGS_EXTRA="-DRAW_LOG_MODE=1"

Usage:
    python log_imu.py --port /dev/cu.SLAB_USBtoUART --seconds 45
    python log_imu.py --port /dev/cu.SLAB_USBtoUART --seconds 45 --out hover_log.csv

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


COLUMNS = ["t_s", "gx_rads", "gy_rads", "gz_rads", "ax_g", "ay_g", "az_g"]


def parse_args():
    p = argparse.ArgumentParser(description="Log IMU noise data from FC firmware")
    p.add_argument("--port", default="/dev/cu.SLAB_USBtoUART",
                   help="Serial port (default: /dev/cu.SLAB_USBtoUART)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=45.0,
                   help="Duration to log in seconds (default: 45)")
    p.add_argument("--out", default=None,
                   help="Output CSV filename (default: imu_log_YYYYMMDD_HHMMSS.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"imu_log_{stamp}.csv"

    out_path = Path(args.out)
    print(f"Port : {args.port} @ {args.baud}")
    print(f"Duration : {args.seconds} s")
    print(f"Output   : {out_path}")
    print()
    print("Waiting for RAW lines — firmware must be built with RAW_LOG_MODE=1")
    print("Press Ctrl-C to stop early.")
    print()

    rows = 0
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
                if not line.startswith("RAW,"):
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    continue

                try:
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    continue

                now = time.monotonic()
                if t_start is None:
                    t_start = now
                    print(f"  First sample received — logging for {args.seconds} s ...", flush=True)

                elapsed = now - t_start
                if elapsed > args.seconds:
                    break

                writer.writerow([f"{elapsed:.4f}"] + [f"{v:.6f}" for v in vals])
                rows += 1

                if rows % 500 == 0:
                    pct = elapsed / args.seconds * 100
                    print(f"  {rows:5d} samples  {elapsed:5.1f} s  ({pct:.0f}%)", flush=True)

        except KeyboardInterrupt:
            print("\nStopped early by user.")

    ser.close()
    print(f"\nDone. {rows} samples saved to {out_path}")
    if rows == 0:
        print("WARNING: No samples captured. Is the firmware built with RAW_LOG_MODE=1?")
    else:
        actual_hz = rows / (time.monotonic() - t_start)
        print(f"Effective sample rate: {actual_hz:.1f} Hz")


if __name__ == "__main__":
    main()
