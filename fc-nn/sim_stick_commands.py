#!/usr/bin/env python3
"""
sim_stick_commands.py — fly the FIRMWARE replica with pilot commands.

Mirrors main.c + rc.c: arm on the ground in HOVER (50-tick ramp), then a
scripted sequence of mode / stick inputs, setpoints slewed exactly like rc.c
(30°/s angle, 0.2 m/s altitude, 0.15 m/s in LAND), fed to
fw_common.FirmwareController (obs[6:7] = attitude − setpoint, P+D on the
commanded altitude).  Physics: test_firmware_sim.Plant (MuJoCo).

Answers "does this policy follow stick commands?" before anything flies.

    python sim_stick_commands.py --model runs/<run>/best_model.zip
    python sim_stick_commands.py --weights ../weights.h --ahrs
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np

from fw_common import FW, Madgwick, BaroModel, FirmwareController, parse_weights_h, load_sb3_weights
from test_firmware_sim import Plant, LOOP_DT
from drone_env import GYRO_NOISE_STD, ACCEL_NOISE_STD

HERE = Path(__file__).resolve().parent


def fc_config(name):
    m = re.search(rf"#define\s+{name}\s+\(?\s*([-+0-9.eE]+)f?", (HERE.parent / "fc_config.h").read_text())
    return float(m.group(1))


RC = {k: fc_config(k) for k in ("RC_MAX_ANGLE_DEG", "RC_ANGLE_RATE_DPS", "RC_ALT_RATE",
                                 "RC_LAND_RATE", "RC_LAND_TARGET")}

# (start_s, label, mode, roll_frac, pitch_frac, alt_m) — roll/pitch as stick fraction
SCHEDULE = [
    (0.0,  "takeoff+hover", "HOVER", 0, 0, 0.3),
    (6.0,  "forward (pitch +full)", "STICK", 0, 1, 0.3),
    (9.0,  "stick centred", "STICK", 0, 0, 0.3),
    (12.0, "right (roll +full)", "STICK", 1, 0, 0.3),
    (15.0, "stick centred", "STICK", 0, 0, 0.3),
    (18.0, "climb to 0.5 m", "STICK", 0, 0, 0.5),
    (23.0, "back to 0.3 m", "STICK", 0, 0, 0.3),
    (28.0, "LAND", "LAND", 0, 0, None),
    (36.0, "end", None, 0, 0, None),
]


def slew(cur, tgt, step):
    return cur + max(-step, min(step, tgt - cur))


def fly(weights, seed=0, use_ahrs=False, thrust_scale=1.0):
    plant = Plant(start_alt=0.0, thrust_scale=thrust_scale, seed=seed, ground_start=True)
    rng = plant.rng
    baro = BaroModel(rng, 0.015, init_alt=0.0)
    ahrs = Madgwick()
    ctrl = FirmwareController(weights, ramp_ticks=FW["RAMP_TICKS"])
    for _ in range(200):                                   # settle on the ground
        g, a = plant.imu(GYRO_NOISE_STD, ACCEL_NOISE_STD)
        ahrs.update(*g, *a, LOOP_DT)
    cmd_r = cmd_p = 0.0
    tgt = FW["TARGET_ALT"]
    log = []
    end = SCHEDULE[-1][0]
    seg = 0
    for t in range(int(end / LOOP_DT)):
        ts = t * LOOP_DT
        while seg + 1 < len(SCHEDULE) and ts >= SCHEDULE[seg + 1][0]:
            seg += 1
        _, label, mode, rf, pf, alt = SCHEDULE[seg]
        want_r, want_p = rf * RC["RC_MAX_ANGLE_DEG"], pf * RC["RC_MAX_ANGLE_DEG"]
        if mode == "LAND":
            want_alt, rate = RC["RC_LAND_TARGET"], RC["RC_LAND_RATE"]
        else:
            want_alt, rate = alt, RC["RC_ALT_RATE"]
        step_a = RC["RC_ANGLE_RATE_DPS"] * LOOP_DT
        cmd_r, cmd_p = slew(cmd_r, want_r, step_a), slew(cmd_p, want_p, step_a)
        tgt = slew(tgt, want_alt, rate * LOOP_DT)

        tr = plant.truth()
        g, a = plant.imu(GYRO_NOISE_STD, ACCEL_NOISE_STD)
        if use_ahrs:
            ahrs.update(*g, *a, LOOP_DT)
            roll, pitch = ahrs.roll_pitch_deg()
        else:
            roll, pitch = tr["roll"], tr["pitch"]
        baro.update(tr["z"])
        motor, _ = ctrl.step(g, a, roll, pitch, baro.alt, baro.vvel, cmd_r, cmd_p, tgt)
        plant.step(motor)
        vel = plant.env._data.qvel
        log.append(dict(t=ts, seg=seg, z=tr["z"], roll=tr["roll"], pitch=tr["pitch"],
                        cmd_r=cmd_r, cmd_p=cmd_p, tgt=tgt, vx=vel[0], vy=vel[1],
                        gz=math.degrees(vel[5])))
        if abs(tr["roll"]) > 60 or abs(tr["pitch"]) > 60:
            return log, f"tilt limit at t={ts:.1f}s ({label})"
    plant.env.close()
    return log, None


def report(log, failure):
    if failure:
        print(f"  ✗ {failure}")
    by_seg = {}
    for r in log:
        by_seg.setdefault(r["seg"], []).append(r)
    print(f"  {'segment':<24}{'cmd R/P':>12}{'got R/P (last 1.5 s)':>24}{'alt tgt→got':>16}"
          f"{'v fwd/right':>14}{'yaw rate':>10}")
    for i, rows in sorted(by_seg.items()):
        tail = [r for r in rows if r["t"] >= rows[-1]["t"] - 1.5]
        m = lambda k: float(np.mean([r[k] for r in tail]))
        # body forward ≈ world +x for small yaw; nose-down (+pitch) → +x
        print(f"  {SCHEDULE[i][1]:<24}{m('cmd_r'):+5.1f}/{m('cmd_p'):+4.1f}°"
              f"{m('roll'):+12.1f}/{m('pitch'):+5.1f}°"
              f"{m('tgt'):>9.2f}→{m('z'):.2f}m"
              f"{m('vx'):+7.2f}/{-m('vy'):+5.2f}"
              f"{m('gz'):+8.1f}°/s")
    if not failure and log:
        print(f"  landed: z = {log[-1]['z']:.3f} m above ground, attitude "
              f"{log[-1]['roll']:+.1f}/{log[-1]['pitch']:+.1f}°")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--weights")
    ap.add_argument("--ahrs", action="store_true", help="Madgwick attitude instead of truth")
    ap.add_argument("--thrust-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    w = load_sb3_weights(args.model) if args.model else parse_weights_h(args.weights)
    print(f"stick-command sim — {args.model or args.weights}  "
          f"({'Madgwick' if args.ahrs else 'truth'} attitude, thrust ×{args.thrust_scale})")
    report(*fly(w, args.seed, args.ahrs, args.thrust_scale))


if __name__ == "__main__":
    main()
