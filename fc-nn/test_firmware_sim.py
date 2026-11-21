#!/usr/bin/env python3
"""
test_firmware_sim.py — LAYER 2: Fly the FIRMWARE (not the policy) in MuJoCo.

validate.py tests the trained policy under TRAINING conditions. This tests
what you actually flash: weights.h + main.c's exact processing chain
(Madgwick attitude, obs[8] clamp, P+D altitude loop, adaptive IIR, residual
mixer) — under a ladder of increasingly realistic conditions. The scenario
where hover first breaks is your dominant sim-to-real gap.

Scenario ladder:
  A  trained-conditions baseline   ground-truth obs, EMA(0.3), NN does
                                   altitude (obs[8] unclamped), target 1.0 m
  B  firmware control path         + obs[8] clamp ±0.05, P+D alt loop,
                                   adaptive IIR, target 0.3 m (truth attitude)
  C  + Madgwick AHRS               attitude estimated from noisy IMU
  D  + real measured noise         gyro/accel noise from flight_test.csv
  E  + thrust deficit 25%          real hover point above sim's linear map
  F  + loop stalls                 20 ms UART-style stall every 20 ticks

Each scenario: 5 runs × up to 30 s. Reports survival, altitude/attitude RMS,
NN saturation — compare the failure signature against your real flight
(NN std ≈ 0.35 rail-to-rail, motors ≈ 1364 µs mean, roll/pitch std ≈ 8°).

Usage:  .venv/bin/python test_firmware_sim.py [--weights ../weights.h]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import mujoco

from drone_env import (
    QuadrotorEnv, BASE_MASS, MAX_THRUST, HOVER_THROTTLE, ACTION_SCALE,
    KM_KF_RATIO, MOTOR_DIRS, GRAVITY, FRAMESKIP, DT,
    GYRO_NOISE_STD, ACCEL_NOISE_STD,
    _quat_to_euler, _quat_rotate_vec,
)
from fw_common import (
    FW, parse_weights_h, nn_forward_np, Madgwick, FirmwareController,
    measure_flight_noise,
)

HERE = Path(__file__).parent
LOOP_DT = FW["LOOP_DT"]


# ---------------------------------------------------------------------------
# Bare-physics wrapper (no env.step smoothing/setpoints — firmware territory)
# ---------------------------------------------------------------------------
class Plant:
    def __init__(self, start_alt: float, thrust_scale: float = 1.0,
                 seed: int = 0):
        self.env = QuadrotorEnv(render_mode=None)
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
        cr, sr = math.cos(tx / 2), math.sin(tx / 2)
        cp, sp = math.cos(ty / 2), math.sin(ty / 2)
        e._data.qpos[0:3] = [0, 0, start_alt]
        e._data.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
        for aid in e._act_ids:
            e._model.actuator_gear[aid, 2] = MAX_THRUST * thrust_scale
        mujoco.mj_forward(e._model, e._data)

    def step(self, motor_cmd, dt_ticks: int = 1):
        e = self.env
        motor_cmd = np.asarray(motor_cmd, dtype=np.float64)
        thrusts = motor_cmd * e._max_thrust
        e._data.ctrl[e._act_ids] = motor_cmd
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
        return dict(z=pos[2], vz=vel[2],
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
# Baro simulator ≈ baro.c (25 Hz decimation, EMA vvel with VVEL_ALPHA)
# ---------------------------------------------------------------------------
class BaroSim:
    def __init__(self, rng, noise_std=0.05):
        self.rng = rng
        self.noise = noise_std
        self.alt = None
        self.vvel = 0.0
        self._tick = 0

    def update(self, true_z):
        self._tick += 1
        if self._tick % FW["BARO_DECIMATION"] != 0:
            return
        meas = true_z + self.rng.normal(0, self.noise)
        if self.alt is None:
            self.alt = meas
            return
        prev = self.alt
        self.alt = 0.4 * meas + 0.6 * self.alt          # BMP280 IIR-ish
        v_raw = (self.alt - prev) / (FW["BARO_DECIMATION"] * LOOP_DT)
        self.vvel = (FW["VVEL_ALPHA"] * v_raw
                     + (1 - FW["VVEL_ALPHA"]) * self.vvel)


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(weights, *, seed, duration_s=30.0,
                target_alt, use_fw_path, use_ahrs,
                gyro_noise, accel_noise, accel_bias_mag=0.0,
                thrust_scale=1.0, stall_every=0, baro_noise=0.05):
    plant = Plant(start_alt=target_alt, thrust_scale=thrust_scale, seed=seed)
    rng = plant.rng
    baro = BaroSim(rng, baro_noise)
    ahrs = Madgwick()
    accel_bias = rng.normal(0, accel_bias_mag, 3) if accel_bias_mag else np.zeros(3)

    ctrl = FirmwareController(
        weights,
        use_adaptive_iir=use_fw_path,
        clamp_alt_obs=use_fw_path,
        use_alt_pid=use_fw_path,
    )
    # In trained-baseline mode the NN owns altitude around its own target
    if not use_fw_path:
        saved_target = FW["TARGET_ALT"]
        FW["TARGET_ALT"] = target_alt

    n_ticks = int(duration_s / LOOP_DT)
    log = dict(z=[], roll=[], pitch=[], nn=[], mot=[])
    alive = n_ticks

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
            alt = baro.alt if baro.alt is not None else truth["z"]

            motor, dbg = ctrl.step(g, a, roll, pitch, alt, baro.vvel)

            # physics advances 2 extra ticks during a 20 ms UART stall
            plant.step(motor, dt_ticks=3 if stalled else 1)

            log["z"].append(truth["z"])
            log["roll"].append(truth["roll"])
            log["pitch"].append(truth["pitch"])
            log["nn"].append(dbg["nn_smoothed"])
            log["mot"].append(motor)

            if (truth["z"] < 0.03 or abs(truth["roll"]) > 90
                    or abs(truth["pitch"]) > 90):
                alive = t
                break
    finally:
        if not use_fw_path:
            FW["TARGET_ALT"] = saved_target
    plant.env.close()

    z = np.array(log["z"]); nn = np.array(log["nn"]); mot = np.array(log["mot"])
    roll = np.array(log["roll"]); pitch = np.array(log["pitch"])
    return dict(
        survived=alive == n_ticks,
        alive_s=alive * LOOP_DT,
        alt_rmse=float(np.sqrt(np.mean((z - target_alt) ** 2))) if len(z) else 99,
        att_rms=float(np.sqrt(np.mean(roll ** 2 + pitch ** 2))) if len(z) else 99,
        nn_std=float(nn.std()) if len(z) else 0,
        nn_sat=float(np.mean(np.abs(nn) > 0.85)) if len(z) else 0,
        mot_mean_us=float(1000 + mot.mean() * 1000) if len(z) else 0,
    )


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "weights.h"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    weights = parse_weights_h(args.weights)
    print(f"═══ LAYER 2: firmware-in-the-loop sim — {args.weights} ═══\n")

    # measured noise, if a flight log exists
    csv = HERE / "flight_test.csv"
    if csv.exists():
        m = measure_flight_noise(csv)
        real_g, real_a = m["gyro_std"], m["accel_std"]
        print(f"Using measured flight noise: gyro {real_g:.3f} rad/s, "
              f"accel {real_a:.3f} g  (from {csv.name})\n")
    else:
        real_g, real_a = 0.25, 0.90
        print(f"No flight log found — using fallback noise "
              f"gyro {real_g}, accel {real_a}\n")

    scenarios = [
        ("A trained baseline",  dict(target_alt=1.0, use_fw_path=False,
                                     use_ahrs=False,
                                     gyro_noise=GYRO_NOISE_STD,
                                     accel_noise=ACCEL_NOISE_STD)),
        ("B firmware path",     dict(target_alt=0.3, use_fw_path=True,
                                     use_ahrs=False,
                                     gyro_noise=GYRO_NOISE_STD,
                                     accel_noise=ACCEL_NOISE_STD)),
        ("C + Madgwick AHRS",   dict(target_alt=0.3, use_fw_path=True,
                                     use_ahrs=True,
                                     gyro_noise=GYRO_NOISE_STD,
                                     accel_noise=ACCEL_NOISE_STD)),
        ("D + real noise",      dict(target_alt=0.3, use_fw_path=True,
                                     use_ahrs=True,
                                     gyro_noise=real_g, accel_noise=real_a,
                                     accel_bias_mag=0.05)),
        ("E + thrust deficit",  dict(target_alt=0.3, use_fw_path=True,
                                     use_ahrs=True,
                                     gyro_noise=real_g, accel_noise=real_a,
                                     accel_bias_mag=0.05,
                                     thrust_scale=0.75)),
        ("F + loop stalls",     dict(target_alt=0.3, use_fw_path=True,
                                     use_ahrs=True,
                                     gyro_noise=real_g, accel_noise=real_a,
                                     accel_bias_mag=0.05,
                                     stall_every=20)),
    ]

    print(f"{'scenario':<22}{'surv':>6}{'alive':>8}{'altRMSE':>9}"
          f"{'attRMS':>8}{'nnStd':>7}{'nnSat%':>8}{'motMean':>9}")
    print("─" * 77)

    results = {}
    for name, kw in scenarios:
        runs = [run_episode(weights, seed=100 + i,
                            duration_s=args.duration, **kw)
                for i in range(args.runs)]
        surv = sum(r["survived"] for r in runs)
        avg = lambda k: float(np.mean([r[k] for r in runs]))
        results[name] = runs
        print(f"{name:<22}{surv:>3}/{args.runs:<2}"
              f"{avg('alive_s'):>7.1f}s"
              f"{avg('alt_rmse'):>8.2f}m"
              f"{avg('att_rms'):>7.1f}°"
              f"{avg('nn_std'):>7.2f}"
              f"{avg('nn_sat')*100:>7.1f}%"
              f"{avg('mot_mean_us'):>8.0f}µ")

    print("\nReference — your REAL flight signature (flight_test.csv):")
    print("  nn_std ≈ 0.35, rail-to-rail; motors ≈ 1364 µs mean; "
          "roll/pitch std 8-9°; could not hold altitude")
    print("\nInterpretation:")
    print("  A fails      → the trained policy itself is broken. Retrain.")
    print("  B fails      → obs[8] clamp / P-alt loop / adaptive IIR broke a")
    print("                 working policy. Fix the firmware control path.")
    print("  C fails      → Madgwick lag: NN needs attitude the estimator")
    print("                 can't deliver. Add AHRS-lag noise in training.")
    print("  D fails      → vibration. Fix accel hardware LPF + retrain with")
    print("                 realistic noise.")
    print("  E fails      → thrust-map mismatch: measure real hover PWM")
    print("                 (TEST 14) and fix NN_HOVER_THROTTLE.")
    print("  F fails      → the blocking-UART stall is destabilising you.")
    return 0


if __name__ == "__main__":
    main()
