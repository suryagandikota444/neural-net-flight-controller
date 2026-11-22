#!/usr/bin/env python3
"""
test_firmware_sim.py — LAYER 2: Fly the FIRMWARE (not the policy) in MuJoCo.

validate.py tests the trained policy under TRAINING conditions. This tests
what you actually flash: weights.h + main.c's exact processing chain
(Madgwick attitude, obs[8] clamp, P+D altitude loop, EMA filter, residual
mixer) — under a ladder of increasingly realistic conditions. The scenario
where hover first breaks is your dominant sim-to-real gap.

Scenario ladder:
  A  trained-conditions baseline   ground-truth obs, EMA(0.3), NN does
                                   altitude (obs[8] unclamped), target 1.0 m
  B  firmware control path         + obs[8] clamp ±0.05, P+D alt loop,
                                   EMA 0.3 filter, target 0.3 m (truth attitude)
  C  + Madgwick AHRS               attitude estimated from noisy IMU
  D  + real measured noise         gyro/accel noise from flight_test.csv
  E  + thrust deficit 25%          real hover point above sim's linear map
  F  + loop stalls                 20 ms UART-style stall every 20 ticks
  G  ground takeoff                C conditions, armed on the ground with
                                   main.c's 50-tick throttle ramp

Altitudes are metres above the barometer's ground reference (the height the
sim body rests at on the floor, drone_env.GROUND_REST_Z), exactly like baro.c.
The barometer is fw_common.BaroModel (baro.c: raw 25 Hz altitude + EMA vvel);
default noise 3 cm σ (bench measured 1.4 cm σ — see drone_env.py).

Each scenario: 5 runs × up to 30 s. Reports survival, altitude/attitude RMS,
NN saturation — compare the failure signature against your real flight
(NN std ≈ 0.35 rail-to-rail, motors ≈ 1364 µs mean, roll/pitch std ≈ 8°).

Usage:  python test_firmware_sim.py [--weights ../weights.h | --model run/best_model.zip]
                                   [--scenarios B,C,D]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import mujoco

from drone_env import (
    QuadrotorEnv, BASE_MASS, MAX_THRUST, HOVER_THROTTLE, ACTION_SCALE, thrust_fraction,
    KM_KF_RATIO, MOTOR_DIRS, GRAVITY, FRAMESKIP, DT,
    GYRO_NOISE_STD, ACCEL_NOISE_STD, GROUND_REST_Z, LIFTOFF_AGL,
    _quat_to_euler, _quat_rotate_vec,
)
from fw_common import (
    FW, parse_weights_h, load_sb3_weights, nn_forward_np, Madgwick,
    FirmwareController, BaroModel, measure_flight_noise,
)

HERE = Path(__file__).parent
LOOP_DT = FW["LOOP_DT"]


# ---------------------------------------------------------------------------
# Bare-physics wrapper (no env.step smoothing/setpoints — firmware territory)
# ---------------------------------------------------------------------------
class Plant:
    """start_alt is metres above the ground reference (GROUND_REST_Z)."""
    def __init__(self, start_alt: float, thrust_scale: float = 1.0,
                 seed: int = 0, ground_start: bool = False):
        # Legacy env: everything below is set by hand, the contract layer is
        # firmware territory (FirmwareController / BaroModel) here.
        self.env = QuadrotorEnv(render_mode=None, firmware_contract=False)
        self.rng = np.random.default_rng(seed)
        e = self.env
        e._randomize_setpoints = False
        e.reset(seed=seed, options={"domain_randomization": False,
            "randomize_setpoints": False, "setpoint": {"alt_m": start_alt}})
        e._max_thrust = np.full(4, MAX_THRUST * thrust_scale)
        e._km_kf = np.full(4, KM_KF_RATIO)
        e._wind_force = np.zeros(3)
        e._model.body_mass[e._drone_body_id] = BASE_MASS
        e._model.body_ipos[e._drone_body_id] = [0, 0, 0]
        mujoco.mj_resetData(e._model, e._data)
        # small random initial tilt, like a real takeoff surface
        tx = self.rng.uniform(-3, 3) * math.pi / 180
        ty = self.rng.uniform(-3, 3) * math.pi / 180
        if ground_start:              # resting level on the floor
            tx = ty = 0.0
            start_alt = 0.0
        cr, sr = math.cos(tx / 2), math.sin(tx / 2)
        cp, sp = math.cos(ty / 2), math.sin(ty / 2)
        e._data.qpos[0:3] = [0, 0, GROUND_REST_Z + start_alt]
        e._data.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
        for aid in e._act_ids:
            e._model.actuator_gear[aid, 2] = MAX_THRUST * thrust_scale
        mujoco.mj_forward(e._model, e._data)

    def step(self, motor_cmd, dt_ticks: int = 1):
        e = self.env
        motor_cmd = np.asarray(motor_cmd, dtype=np.float64)
        frac = thrust_fraction(motor_cmd)        # measured curve (real drone)
        thrusts = frac * e._max_thrust
        e._data.ctrl[e._act_ids] = frac
        yaw_torque = float(np.sum(MOTOR_DIRS * e._km_kf * thrusts))
        quat = e._data.qpos[3:7].copy()
        tw = _quat_rotate_vec(quat, np.array([0.0, 0.0, yaw_torque]))
        e._data.xfrc_applied[e._drone_body_id, 3:6] = tw
        for _ in range(FRAMESKIP * dt_ticks):
            mujoco.mj_step(e._model, e._data)

    # ---- ground truth ----
    def truth(self):
        e = self.env
        pos = e._sensor(e._pos_sensor_id, 3)
        quat = e._sensor(e._quat_sensor_id, 4)
        vel = e._sensor(e._vel_sensor_id, 3)
        r, p, _ = _quat_to_euler(quat)
        return dict(z=pos[2] - GROUND_REST_Z, vz=vel[2],   # z: above ground ref
                    x=pos[0], y=pos[1],
                    roll=math.degrees(r), pitch=math.degrees(p))

    # ---- sensors ----
    def imu(self, gyro_noise, accel_noise, accel_bias=np.zeros(3)):
        e = self.env
        gyro = e._sensor(e._gyro_sensor_id, 3) \
            + self.rng.normal(0, gyro_noise, 3)
        accel = e._sensor(e._accel_sensor_id, 3) / GRAVITY \
            + self.rng.normal(0, accel_noise, 3) + accel_bias
        return gyro, accel


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(weights, *, seed, duration_s=30.0,
                target_alt, use_fw_path, use_ahrs,
                gyro_noise, accel_noise, accel_bias_mag=0.0,
                thrust_scale=1.0, stall_every=0, baro_noise=0.03,
                ground_start=False):
    plant = Plant(start_alt=target_alt, thrust_scale=thrust_scale, seed=seed,
                  ground_start=ground_start)
    rng = plant.rng
    # baro.c has been running since power-on: start settled at the true height
    baro = BaroModel(rng, baro_noise, init_alt=plant.truth()["z"])
    ahrs = Madgwick()
    accel_bias = rng.normal(0, accel_bias_mag, 3) if accel_bias_mag else np.zeros(3)

    ctrl = FirmwareController(
        weights,
        use_adaptive_iir=False,   # main.c uses the training EMA since 2026-11-21
        clamp_alt_obs=use_fw_path,
        use_alt_pid=use_fw_path,
        ramp_ticks=FW["RAMP_TICKS"] if ground_start else 0,
    )
    # In trained-baseline mode the NN owns altitude around its own target
    if not use_fw_path:
        saved_target = FW["TARGET_ALT"]
        FW["TARGET_ALT"] = target_alt

    n_ticks = int(duration_s / LOOP_DT)
    log = dict(z=[], roll=[], pitch=[], nn=[], mot=[], xy=[])
    alive = n_ticks
    airborne = not ground_start

    # Let AHRS converge on the ground (mirrors main.c 2 s settle)
    for _ in range(200):
        g, a = plant.imu(gyro_noise, accel_noise, accel_bias)
        ahrs.update(*g, *a, LOOP_DT)

    try:
        for t in range(n_ticks):
            truth = plant.truth()
            g, a = plant.imu(gyro_noise, accel_noise, accel_bias)

            # dt for this tick (loop stall model: AHRS still assumes 10 ms)
            stalled = stall_every and (t % stall_every == stall_every - 1)

            if use_ahrs:
                ahrs.update(*g, *a, LOOP_DT)    # firmware uses FIXED dt
                roll, pitch = ahrs.roll_pitch_deg()
            else:
                roll, pitch = truth["roll"], truth["pitch"]

            baro.update(truth["z"])
            motor, dbg = ctrl.step(g, a, roll, pitch, baro.alt, baro.vvel)

            # physics advances 2 extra ticks during a 20 ms UART stall
            plant.step(motor, dt_ticks=3 if stalled else 1)

            log["z"].append(truth["z"])
            log["roll"].append(truth["roll"])
            log["pitch"].append(truth["pitch"])
            log["nn"].append(dbg["nn_smoothed"])
            log["mot"].append(motor)
            log["xy"].append((truth["x"], truth["y"]))

            airborne = airborne or truth["z"] > LIFTOFF_AGL
            if ((airborne and truth["z"] < 0.03) or abs(truth["roll"]) > 90
                    or abs(truth["pitch"]) > 90):
                alive = t
                break
    finally:
        if not use_fw_path:
            FW["TARGET_ALT"] = saved_target
    plant.env.close()

    z = np.array(log["z"]); nn = np.array(log["nn"]); mot = np.array(log["mot"])
    roll = np.array(log["roll"]); pitch = np.array(log["pitch"])
    xy = np.array(log["xy"]).reshape(-1, 2)
    if ground_start and len(z):          # score the hover, not the climb
        k = len(z) // 3
        z, nn, mot, roll, pitch = z[k:], nn[k:], mot[k:], roll[k:], pitch[k:]
    if not airborne:
        alive = 0                        # never left the ground
    return dict(
        survived=alive == n_ticks,
        mean_roll=float(roll.mean()) if len(z) else 99,
        mean_pitch=float(pitch.mean()) if len(z) else 99,
        max_drift_m=float(np.max(np.linalg.norm(xy, axis=1))) if len(xy) else 0,
        alive_s=alive * LOOP_DT,
        alt_rmse=float(np.sqrt(np.mean((z - target_alt) ** 2))) if len(z) else 99,
        att_rms=float(np.sqrt(np.mean(roll ** 2 + pitch ** 2))) if len(z) else 99,
        nn_std=float(nn.std()) if len(z) else 0,
        nn_sat=float(np.mean(np.abs(nn) > 0.85)) if len(z) else 0,
        mot_mean_us=float(1000 + mot.mean() * 1000) if len(z) else 0,
    )


# ---------------------------------------------------------------------------
def flight_noise():
    """Measured in-flight IMU noise if flight_test.csv exists, else fallback."""
    csv = HERE / "flight_test.csv"
    if csv.exists():
        m = measure_flight_noise(csv)
        return m["gyro_std"], m["accel_std"], csv.name
    return 0.25, 0.90, None


def scenario_ladder(real_g, real_a):
    """Name → run_episode kwargs. target_alt is metres above ground ref."""
    fw = dict(target_alt=FW["TARGET_ALT"], use_fw_path=True)
    real = dict(gyro_noise=real_g, accel_noise=real_a, accel_bias_mag=0.05)
    train = dict(gyro_noise=GYRO_NOISE_STD, accel_noise=ACCEL_NOISE_STD)
    return {
        "A trained baseline":  dict(target_alt=1.0, use_fw_path=False,
                                    use_ahrs=False, **train),
        "B firmware path":     dict(**fw, use_ahrs=False, **train),
        "C + Madgwick AHRS":   dict(**fw, use_ahrs=True, **train),
        "D + real noise":      dict(**fw, use_ahrs=True, **real),
        "E + thrust deficit":  dict(**fw, use_ahrs=True, **real, thrust_scale=0.75),
        "F + loop stalls":     dict(**fw, use_ahrs=True, **real, stall_every=20),
        "G ground takeoff":    dict(**fw, use_ahrs=True, **train, ground_start=True),
    }


def run_ladder(weights, letters="ABCDEFG", runs=5, duration=30.0, seed0=100,
               verbose=True):
    real_g, real_a, src = flight_noise()
    if verbose:
        print(f"IMU noise for D–F: gyro {real_g:.3f} rad/s, accel {real_a:.3f} g "
              f"({'measured from ' + src if src else 'fallback'})\n")
        print(f"{'scenario':<22}{'surv':>6}{'alive':>8}{'altRMSE':>9}"
              f"{'attRMS':>8}{'meanR':>7}{'meanP':>7}{'nnStd':>7}{'nnSat%':>8}"
              f"{'motMean':>9}{'drift':>7}")
        print("─" * 98)
    results = {}
    for name, kw in scenario_ladder(real_g, real_a).items():
        if name[0] not in letters:
            continue
        rs = [run_episode(weights, seed=seed0 + i, duration_s=duration, **kw)
              for i in range(runs)]
        results[name] = rs
        if verbose:
            surv = sum(r["survived"] for r in rs)
            avg = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"{name:<22}{surv:>3}/{runs:<2}"
                  f"{avg('alive_s'):>7.1f}s"
                  f"{avg('alt_rmse'):>8.2f}m"
                  f"{avg('att_rms'):>7.1f}°"
                  f"{avg('mean_roll'):>6.1f}°"
                  f"{avg('mean_pitch'):>6.1f}°"
                  f"{avg('nn_std'):>7.2f}"
                  f"{avg('nn_sat')*100:>7.1f}%"
                  f"{avg('mot_mean_us'):>8.0f}µ"
                  f"{avg('max_drift_m'):>6.1f}m", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--weights", default=str(HERE.parent / "weights.h"))
    src.add_argument("--model", help="SB3 .zip instead of a weights.h")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--scenarios", default="ABCDEFG",
                    help="letters to run, e.g. BCD")
    args = ap.parse_args()

    weights = load_sb3_weights(args.model) if args.model else parse_weights_h(args.weights)
    print(f"═══ LAYER 2: firmware-in-the-loop sim — {args.model or args.weights} ═══\n")
    run_ladder(weights, args.scenarios.replace(",", "").upper(),
               args.runs, args.duration)

    print("\nReference — your REAL flight signature (flight_test.csv):")
    print("  nn_std ≈ 0.35, rail-to-rail; motors ≈ 1364 µs mean; "
          "roll/pitch std 8-9°; could not hold altitude")
    print("\nInterpretation:")
    print("  A fails      → (legacy-trained policies only: NN owns altitude at")
    print("                 1 m, true hvel) the trained policy itself is broken.")
    print("  B fails      → obs[8] clamp / P-alt loop / obs[10:12]=0 broke the")
    print("                 policy. Retrain with firmware_contract=True.")
    print("  C fails      → Madgwick lag: NN needs attitude the estimator")
    print("                 can't deliver. Add AHRS-lag noise in training.")
    print("  D fails      → vibration. Fix accel hardware LPF + retrain with")
    print("                 realistic noise.")
    print("  E fails      → thrust-map mismatch: measure real hover PWM")
    print("                 (TEST 14) and fix NN_HOVER_THROTTLE.")
    print("  F fails      → the blocking-UART stall is destabilising you.")
    print("  G fails      → takeoff (ramp + ground-effect-free climb) is unstable.")
    return 0


if __name__ == "__main__":
    main()
