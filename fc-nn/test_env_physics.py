#!/usr/bin/env python3
"""
test_env_physics.py — LAYER 1: Is the simulator physically sane?

If the sim is wrong, the NN learned to fly the wrong aircraft and nothing
downstream can fix that. Each test manipulates the MuJoCo model directly
(no policy involved) and checks first-principles physics:

  1.  Hover equilibrium — commanding exactly HOVER_THROTTLE holds altitude.
  2.  Accelerometer convention — az obs = +0.5 at rest (gravity +1g / 2g).
  3.  Gyro obs scaling & sign.
  4.  Mixer geometry — front pair ↑ ⇒ nose up torque, left pair ↑ ⇒ roll,
      with signs matching the euler convention the obs use.
  5.  Yaw reaction torque sign vs MOTOR_DIRS.
  6.  Thrust-to-weight ratio report.
  7.  Training-noise vs real-flight-noise coverage (uses flight_test.csv).

Usage:  .venv/bin/python test_env_physics.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import mujoco

from drone_env import (
    QuadrotorEnv, BASE_MASS, ACTION_SCALE,
    LEGACY_MAX_THRUST as MAX_THRUST, LEGACY_HOVER_THROTTLE as HOVER_THROTTLE,  # legacy env, linear map
    KM_KF_RATIO, MOTOR_DIRS, GRAVITY, FRAMESKIP,
    GYRO_NOISE_STD, ACCEL_NOISE_STD, _quat_to_euler, _quat_rotate_vec,
)
from fw_common import measure_flight_noise

HERE = Path(__file__).parent
PASS = "  \033[32m✓ PASS\033[0m"
FAIL = "  \033[31m✗ FAIL\033[0m"
WARN = "  \033[33m! WARN\033[0m"
failures = 0


def check(cond, msg):
    global failures
    print(f"{PASS if cond else FAIL}  {msg}")
    failures += not cond


def make_clean_env():
    """Env with zero domain randomisation and known parameters.
    Legacy mode: these checks start at z = 1 m via the altitude setpoint
    (physics is identical in both modes; raw_step bypasses the contract)."""
    env = QuadrotorEnv(render_mode=None, firmware_contract=False)
    env.reset(seed=0, options={"domain_randomization": False,
        "randomize_setpoints": False, "setpoint": {"alt_m": 1.0},
        "initial_rpy_deg": [0,0,0], "sensor_noise": False})
    return env


def raw_step(env, motor_cmd, n_policy_steps=1):
    """Step physics with direct motor commands — NO action smoothing,
    NO setpoint logic. This is what the firmware actually does."""
    motor_cmd = np.asarray(motor_cmd, dtype=np.float64)
    for _ in range(n_policy_steps):
        thrusts = motor_cmd * env._max_thrust
        env._data.ctrl[env._act_ids] = motor_cmd
        yaw_torque = float(np.sum(MOTOR_DIRS * env._km_kf * thrusts))
        quat = env._data.qpos[3:7].copy()
        tw = _quat_rotate_vec(quat, np.array([0.0, 0.0, yaw_torque]))
        env._data.xfrc_applied[env._drone_body_id, 0:3] = env._wind_force
        env._data.xfrc_applied[env._drone_body_id, 3:6] = tw
        for _ in range(FRAMESKIP):
            mujoco.mj_step(env._model, env._data)


def state(env):
    pos = env._sensor(env._pos_sensor_id, 3)
    quat = env._sensor(env._quat_sensor_id, 4)
    vel = env._sensor(env._vel_sensor_id, 3)
    angvel = env._sensor(env._angvel_sensor_id, 3)
    r, p, y = _quat_to_euler(quat)
    return dict(z=pos[2], roll=math.degrees(r), pitch=math.degrees(p),
                yaw=math.degrees(y), vz=vel[2], angvel=angvel)


def main():
    print("═══ LAYER 1: Simulation physics sanity ═══\n")

    # ── 1. Hover equilibrium ─────────────────────────────────────
    env = make_clean_env()
    raw_step(env, np.full(4, HOVER_THROTTLE), n_policy_steps=200)  # 2 s
    s = state(env)
    check(abs(s["z"] - 1.0) < 0.05,
          f"hover equilibrium: z = {s['z']:.4f} m after 2 s at cmd="
          f"{HOVER_THROTTLE:.4f} (drift {abs(s['z']-1)*100:.1f} cm, want <5 cm)")
    check(abs(s["vz"]) < 0.05,
          f"hover equilibrium: vz = {s['vz']:+.4f} m/s (want ~0)")

    # ── 2. Accelerometer convention at rest ─────────────────────
    env = make_clean_env()
    raw_step(env, np.full(4, HOVER_THROTTLE), n_policy_steps=10)
    accel = env._sensor(env._accel_sensor_id, 3) / GRAVITY
    check(abs(accel[2] - 1.0) < 0.05 and abs(accel[0]) < 0.05,
          f"accel at hover = [{accel[0]:+.3f}, {accel[1]:+.3f}, "
          f"{accel[2]:+.3f}] g (want ≈ [0,0,+1] → obs az = +0.5)")

    # ── 3. Gyro scaling & sign ───────────────────────────────────
    env = make_clean_env()
    env._data.qvel[3] = 1.0            # +1 rad/s body roll rate
    mujoco.mj_forward(env._model, env._data)
    gyro = env._sensor(env._gyro_sensor_id, 3)
    check(abs(gyro[0] - 1.0) < 0.05,
          f"gyro reads applied +1 rad/s about X: gx = {gyro[0]:+.4f}")

    # ── 4. Mixer geometry ────────────────────────────────────────
    # Motors at +X arms (FL=0, FR=1): thrust ↑ front ⇒ torque about -Y
    # ⇒ with euler pitch = asin(2(wy−zx)), pitch goes NEGATIVE (nose up in
    # this convention is negative pitch? let physics decide — we assert the
    # DIRECTION the obs will see, which is what the NN was trained on).
    env = make_clean_env()
    cmd = np.full(4, HOVER_THROTTLE); cmd[[0, 1]] += 0.08   # front pair up
    raw_step(env, cmd, n_policy_steps=20)
    s_front = state(env)
    check(s_front["pitch"] < -1.0,
          f"front pair +0.08 ⇒ pitch = {s_front['pitch']:+.2f}° "
          f"(negative = nose rises in trained convention)")

    env = make_clean_env()
    cmd = np.full(4, HOVER_THROTTLE); cmd[[0, 3]] += 0.08   # left pair (FL,RL)
    raw_step(env, cmd, n_policy_steps=20)
    s_left = state(env)
    check(s_left["roll"] > 1.0,
          f"left pair +0.08 ⇒ roll = {s_left['roll']:+.2f}° "
          f"(positive = left side rises, drone tips right)")

    # ── 5. Yaw reaction torque sign ──────────────────────────────
    env = make_clean_env()
    cmd = np.full(4, HOVER_THROTTLE); cmd[[0, 2]] += 0.10   # CCW props (FL,RR)
    raw_step(env, cmd, n_policy_steps=30)
    gz = env._sensor(env._gyro_sensor_id, 3)[2]
    check(gz < -0.01,
          f"CCW motors (FL,RR) up ⇒ body yaws CW: gz = {gz:+.4f} rad/s "
          f"(want negative)")
    print("        NOTE: this requires the REAL props to spin FL=CCW, FR=CW,")
    print("        RR=CCW, RL=CW (viewed from above). If your build follows")
    print("        the Betaflight DEFAULT (M1/RR=CW, M2/FR=CCW, M3/RL=CCW,")
    print("        M4/FL=CW) the yaw axis is INVERTED vs training. Verify")
    print("        physically with props off — spin direction, not just order.")

    # ── 6. Thrust-to-weight ─────────────────────────────────────
    twr = 4 * MAX_THRUST / (BASE_MASS * GRAVITY)
    print(f"\n  thrust-to-weight (sim): {twr:.2f}  "
          f"(hover at {HOVER_THROTTLE:.3f} cmd = "
          f"{1000+HOVER_THROTTLE*1000:.0f} µs — LINEAR thrust map!)")
    print("  Real ESCs are NOT linear in pulse width. The real hover PWM is")
    print("  unknown until you run firmware TEST 14 (throttle staircase).")

    # ── 7. Does training noise cover reality? ───────────────────
    print()
    csv = HERE / "flight_test.csv"
    if csv.exists():
        m = measure_flight_noise(csv)
        g_ratio = m["gyro_std"] / GYRO_NOISE_STD
        a_ratio = m["accel_std"] / ACCEL_NOISE_STD
        print(f"  measured in-flight noise (from {csv.name}):")
        print(f"    gyro  std ≈ {m['gyro_std']:.3f} rad/s  vs trained "
              f"{GYRO_NOISE_STD} ({g_ratio:.1f}x)")
        print(f"    accel std ≈ {m['accel_std']:.3f} g      vs trained "
              f"{ACCEL_NOISE_STD} ({a_ratio:.1f}x)")
        check(g_ratio < 1.5,
              f"training gyro noise covers reality ({g_ratio:.1f}x)")
        check(a_ratio < 1.5,
              f"training accel noise covers reality ({a_ratio:.1f}x) — "
              f"if FAIL: NN inputs in flight are OUT OF DISTRIBUTION")
    else:
        print(f"{WARN}  {csv.name} not found — skipping noise-coverage check")

    print(f"\n{'═'*60}\nLayer 1 result: "
          f"{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
