"""
validate.py  –  Two-phase validation for the trained quadrotor model.

Phase 1 – Python forward-pass cross-check
    Runs the network in pure Python/numpy to produce the same output as
    the STM32 C implementation.  Print the raw output so you can feed the
    same observation into your C unit-test and compare byte-for-byte.

Phase 2 – Simulation checkpoints (MuJoCo)
    a) Stable hover for 30 s with no wind
    b) Recovery from a 30° initial tilt within 2 s
    c) Stable hover for 10 s with 0.5 N constant wind

Usage:
    python validate.py runs/quad_hover/best_model.zip
    python validate.py runs/quad_hover/best_model.zip --render   # with 3D viewer
    python validate.py runs/quad_hover/best_model.zip --numpy-only
"""

import argparse
import math
import sys
import numpy as np


# -----------------------------------------------------------------------
# Phase 1 – pure-numpy forward pass (mirrors the STM32 C code exactly)
# -----------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def forward_numpy(obs: np.ndarray, W1, B1, W2, B2, W3, B3) -> np.ndarray:
    """
    Inference using the raw weight arrays – identical to the C firmware loop.

      h1 = relu(W1 @ obs + B1)
      h2 = relu(W2 @ h1  + B2)
      out =     W3 @ h2  + B3
    """
    assert obs.shape == (12,), f"Expected obs shape (12,), got {obs.shape}"
    h1  = relu(W1 @ obs + B1)
    h2  = relu(W2 @ h1  + B2)
    out = W3 @ h2 + B3
    return out.astype(np.float32)


def pwm_from_output(out: np.ndarray) -> np.ndarray:
    """Clip network output to [0,1] then scale to [1000, 2000] µs."""
    clipped = np.clip(out, 0.0, 1.0)
    return (clipped * 1000.0 + 1000.0).astype(np.float32)


def load_weights_from_model(model_path: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    policy_net   = model.policy.mlp_extractor.policy_net
    linear_layers = [m for m in policy_net if hasattr(m, "weight")]
    layer1, layer2 = linear_layers
    layer3 = model.policy.action_net
    W1 = layer1.weight.detach().cpu().numpy().astype(np.float32)
    B1 = layer1.bias.detach().cpu().numpy().astype(np.float32)
    W2 = layer2.weight.detach().cpu().numpy().astype(np.float32)
    B2 = layer2.bias.detach().cpu().numpy().astype(np.float32)
    W3 = layer3.weight.detach().cpu().numpy().astype(np.float32)
    B3 = layer3.bias.detach().cpu().numpy().astype(np.float32)
    return W1, B1, W2, B2, W3, B3, model


def run_forward_pass_check(W1, B1, W2, B2, W3, B3, sb3_model=None):
    print("=" * 60)
    print("PHASE 1 – Python / numpy forward-pass cross-check")
    print("=" * 60)

    obs = np.array([
        0.0,   # gyro_x  / MAX_GYRO
        0.0,   # gyro_y
        0.0,   # gyro_z
        0.0,   # accel_x / 2g
        0.0,   # accel_y / 2g
        0.5,   # accel_z / 2g  (1 g upward -> 0.5)
        0.0,   # roll_error  / 45  (ANGLE_ERR_SCALE)
        0.0,   # pitch_error / 45  (ANGLE_ERR_SCALE)
        0.0,   # altitude_error / ALT_SCALE  (at target = 0)
        0.0,   # vertical_vel  / VVEL_SCALE  (stationary = 0)
        0.0,   # hvel_x / HVEL_SCALE         (no lateral drift)
        0.0,   # hvel_y / HVEL_SCALE
    ], dtype=np.float32)

    print("\nInput observation (same values to feed your C unit-test):")
    for i, v in enumerate(obs):
        print(f"  obs[{i}] = {v:.9e}f")

    out_np  = forward_numpy(obs, W1, B1, W2, B2, W3, B3)
    pwm_np  = pwm_from_output(out_np)

    print("\nNumpy network output (raw, pre-clip):")
    for i, v in enumerate(out_np):
        print(f"  out[{i}] = {v:.9e}f")

    print("\nMotor PWM commands (µs):")
    motor_names = ["Front-Left (CCW)", "Front-Right (CW)",
                   "Rear-Right (CCW)", "Rear-Left  (CW) "]
    for i, (name, pwm) in enumerate(zip(motor_names, pwm_np)):
        print(f"  Motor {i+1} {name}: {pwm:.1f} µs")

    if sb3_model is not None:
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            feats = sb3_model.policy.extract_features(
                obs_t, sb3_model.policy.features_extractor)
            latent_pi, _ = sb3_model.policy.mlp_extractor(feats)
            out_sb3 = sb3_model.policy.action_net(latent_pi).squeeze().cpu().numpy()
        max_diff = np.max(np.abs(out_np - out_sb3))
        print(f"\nMax |numpy - PyTorch| difference: {max_diff:.2e}")
        if max_diff < 1e-5:
            print("✓  Numpy and PyTorch outputs match (within 1e-5).")
        else:
            print("✗  WARNING: outputs differ – check layer extraction.")
    print()


# -----------------------------------------------------------------------
# Phase 2 – Simulation validation checkpoints  (MuJoCo)
# -----------------------------------------------------------------------

def _make_fixed_env(render: bool = False, wind_force=None,
                    init_tilt_deg: float = 0.0, max_steps: int = 3000):
    from drone_env import (QuadrotorEnv, BASE_MASS, MAX_THRUST, KM_KF_RATIO)
    import mujoco

    env = QuadrotorEnv(render_mode="human" if render else None)
    env._max_ep_steps    = max_steps
    env._test_wind_force = wind_force     # mutable – update between tests
    env._test_tilt_deg   = init_tilt_deg  # mutable – update between tests

    def fixed_reset(*, seed=None, options=None):
        import gymnasium as gym
        gym.Env.reset(env, seed=seed)

        env._max_thrust = np.full(4, MAX_THRUST)
        env._km_kf      = np.full(4, KM_KF_RATIO)
        env._wind_force = (np.array(env._test_wind_force, dtype=np.float64)
                           if env._test_wind_force is not None
                           else np.zeros(3))

        mujoco.mj_resetData(env._model, env._data)
        env._model.body_mass[env._drone_body_id] = BASE_MASS
        env._model.body_ipos[env._drone_body_id] = [0, 0, 0]

        tilt = math.radians(env._test_tilt_deg)
        cr, sr = math.cos(tilt / 2), math.sin(tilt / 2)
        quat = np.array([cr, sr, 0.0, 0.0])   # tilt around X

        env._data.qpos[0:3] = [0.0, 0.0, 1.0]
        env._data.qpos[3:7] = quat

        mujoco.mj_forward(env._model, env._data)
        env._step_count  = 0
        env._prev_action = np.zeros(4, dtype=np.float32)
        return env._get_obs(), {}

    env.reset = fixed_reset
    return env


def _run_episode(env, policy_fn, max_steps, pass_fn):
    obs, _ = env.reset()
    pass_steps  = 0
    total_steps = 0
    crashed     = False

    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_steps += 1

        if pass_fn(env):
            pass_steps += 1
        if terminated:
            crashed = True
            break
        if truncated:
            break

        # Optional render
        env.render()

    return {"total_steps": total_steps, "pass_steps": pass_steps,
            "crashed": crashed}


def run_simulation_checkpoints(sb3_model, render: bool = False):
    import torch
    from drone_env import _quat_to_euler

    FRAME_SKIP = 40          # must match drone_env.py FRAME_SKIP
    DT_POLICY  = 0.00025 * FRAME_SKIP   # 0.01 s per policy step

    STEPS_30S = int(30.0 / DT_POLICY)   # 3000 policy steps
    STEPS_10S = int(10.0 / DT_POLICY)   # 1000 policy steps
    STEPS_2S  = int( 2.0 / DT_POLICY)   #  200 policy steps

    def policy(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = sb3_model.predict(obs_t, deterministic=True)
        return np.array(action, dtype=np.float32).flatten()

    print("=" * 60)
    print("PHASE 2 – Simulation checkpoints  (MuJoCo)")
    print("=" * 60)

    results = {}

    # One env / one viewer window shared across all three tests
    env = _make_fixed_env(render=render, max_steps=STEPS_30S)

    # a) Hover 30 s, no wind
    print("\na) Stable hover – 30 s, no wind ...")
    env._max_ep_steps    = STEPS_30S
    env._test_wind_force = None
    env._test_tilt_deg   = 0.0

    def hover_pass(e):
        pos = e._sensor(e._pos_sensor_id, 3)
        return abs(pos[2] - 1.0) < 0.05

    r = _run_episode(env, policy, STEPS_30S, hover_pass)
    pct = 100.0 * r["pass_steps"] / max(r["total_steps"], 1)
    passed = not r["crashed"] and pct >= 90.0
    results["hover_30s"] = passed
    print(f"   Steps: {r['total_steps']:,} / {STEPS_30S:,}")
    print(f"   Within 5 cm: {pct:.1f}%   Crashed: {r['crashed']}")
    print(f"   {'✓ PASS' if passed else '✗ FAIL'}")

    # b) Recovery from 30° tilt
    print("\nb) Recovery from 30° tilt – must level within 2 s ...")
    env._max_ep_steps    = STEPS_2S
    env._test_wind_force = None
    env._test_tilt_deg   = 30.0

    def tilt_pass(e):
        quat = e._sensor(e._quat_sensor_id, 4)
        roll, pitch, _ = _quat_to_euler(quat)
        return abs(math.degrees(roll)) < 10.0 and abs(math.degrees(pitch)) < 10.0

    r = _run_episode(env, policy, STEPS_2S, tilt_pass)
    passed = not r["crashed"] and r["pass_steps"] > 0
    results["recovery_30deg"] = passed
    print(f"   Steps: {r['total_steps']:,} / {STEPS_2S:,}")
    print(f"   Steps within ±10°: {r['pass_steps']:,}")
    print(f"   {'✓ PASS' if passed else '✗ FAIL'}")

    # c) Hover 10 s with wind
    print("\nc) Hover with wind – 10 s, 0.5 N constant ...")
    env._max_ep_steps    = STEPS_10S
    env._test_wind_force = [0.5, 0.0, 0.0]
    env._test_tilt_deg   = 0.0

    def wind_pass(e):
        pos = e._sensor(e._pos_sensor_id, 3)
        return abs(pos[2] - 1.0) < 0.10

    r = _run_episode(env, policy, STEPS_10S, wind_pass)
    pct = 100.0 * r["pass_steps"] / max(r["total_steps"], 1)
    passed = not r["crashed"] and pct >= 80.0
    results["wind_hover_10s"] = passed
    print(f"   Steps: {r['total_steps']:,} / {STEPS_10S:,}")
    print(f"   Within 10 cm: {pct:.1f}%")
    print(f"   {'✓ PASS' if passed else '✗ FAIL'}")

    env.close()  # close once after all tests

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_passed = all(results.values())
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print()
    if all_passed:
        print("All checkpoints passed – model is ready for weight export.")
    else:
        print("Some checkpoints FAILED – continue training before deploying.")
    return all_passed


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Path to the trained model .zip.")
    parser.add_argument("--numpy-only", action="store_true",
                        help="Only run the forward-pass check, skip simulation.")
    parser.add_argument("--render", action="store_true",
                        help="Open MuJoCo 3D viewer during simulation tests.")
    args = parser.parse_args()

    W1, B1, W2, B2, W3, B3, sb3_model = load_weights_from_model(args.model)
    run_forward_pass_check(W1, B1, W2, B2, W3, B3, sb3_model)

    if not args.numpy_only:
        ok = run_simulation_checkpoints(sb3_model, render=args.render)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
