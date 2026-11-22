"""
validate.py  –  Two-phase validation for the trained quadrotor model.

Phase 1 – Python forward-pass cross-check
    Runs the network in pure Python/numpy to produce the same output as
    the STM32 C implementation.  Print the raw output so you can feed the
    same observation into your C unit-test and compare byte-for-byte.

Phase 2 – Simulation checkpoints (MuJoCo)
    a) Stable hover for 30 s with no wind
    b) Both signs of roll/pitch tilt, with sustained recovery and altitude-loss bounds
    c) Wind and randomized hover, with full-duration and horizontal-envelope checks

Usage:
    python validate.py runs/quad_hover/best_model.zip
    python validate.py runs/quad_hover/best_model.zip --render   # with 3D viewer
    python validate.py runs/quad_hover/best_model.zip --numpy-only
    python validate.py runs/quad_hover/best_model.zip --legacy   # pre-contract checkpoints

Phase 2 uses the firmware-contract env and CONTRACT_SCENARIOS by default;
--legacy evaluates in the old env (target 1 m, true hvel) with the old suite.
For the full hardware gate (bench vector, firmware replica) use eval_contract.py.
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
    """Nominal residual mapping, before firmware PD/ramp and filter dynamics."""
    from drone_env import HOVER_THROTTLE, ACTION_SCALE
    command = np.clip(HOVER_THROTTLE + ACTION_SCALE * np.clip(out, -1, 1), 0, 1)
    return (command * 1000.0 + 1000.0).astype(np.float32)


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

    matched = True
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
        matched = bool(np.isfinite(max_diff) and max_diff < 1e-5)
        if matched:
            print("✓  Numpy and PyTorch outputs match (within 1e-5).")
        else:
            print("✗  WARNING: outputs differ – check layer extraction.")
    print()
    return matched


# -----------------------------------------------------------------------
# Phase 2 – Simulation validation checkpoints  (MuJoCo)
# -----------------------------------------------------------------------

def run_simulation_checkpoints(sb3_model, render=False, seeds=range(100, 105), output=None,
                               legacy=False):
    from evaluation import evaluate, SCENARIOS, CONTRACT_SCENARIOS
    from drone_env import QuadrotorEnv
    results = evaluate(lambda obs: sb3_model.predict(obs, deterministic=True)[0],
                       seeds=seeds, output=output,
                       scenarios=SCENARIOS if legacy else CONTRACT_SCENARIOS,
                       env_factory=lambda: QuadrotorEnv(render_mode="human" if render else None,
                                                        firmware_contract=not legacy))
    gated = [r for r in results if not r.get("advisory")]
    passed = sum(r["passed"] for r in gated)
    print(f"\nFixed-suite acceptance ({'legacy' if legacy else 'firmware contract'}): "
          f"{passed}/{len(gated)} gated episodes passed"
          f" (+{sum(r['passed'] for r in results if r.get('advisory'))}/"
          f"{len(results) - len(gated)} advisory)")
    results = gated
    print("This evaluates the simulated policy; hardware/controller equivalence is a separate gate.")
    return passed == len(results)


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
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", help="Write criteria, per-episode results and summary JSON")
    parser.add_argument("--legacy", action="store_true",
                        help="Evaluate in the pre-contract env/suite (old checkpoints).")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    W1, B1, W2, B2, W3, B3, sb3_model = load_weights_from_model(args.model)
    parity_ok = run_forward_pass_check(W1, B1, W2, B2, W3, B3, sb3_model)

    if not args.numpy_only:
        ok = run_simulation_checkpoints(sb3_model, render=args.render,
                seeds=range(args.seed_start, args.seed_start + args.runs), output=args.out,
                legacy=args.legacy)
        sys.exit(0 if ok and parity_ok else 1)
    sys.exit(0 if parity_ok else 1)


if __name__ == "__main__":
    main()
