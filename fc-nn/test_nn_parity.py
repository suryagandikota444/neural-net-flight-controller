#!/usr/bin/env python3
"""
test_nn_parity.py — LAYER 0: Is the network in flash the network you trained?

Checks, in order:
  1. weights.h parses and has the right shapes (1604 params).
  2. Which trained model .zip weights.h actually came from (catches stale
     exports — the #1 silent killer in a train→export→flash pipeline).
  3. NN_HOVER_THROTTLE in weights.h == m*g/(4*T_max) used in training.
  4. Forward pass on golden observation vectors; hover obs should produce
     near-zero residuals (a trained hover policy outputting large residuals
     at perfect hover is broken).
  5. Emits ../host_test/golden_vectors.h so the REAL nn.c can be compiled on
     this Mac and checked against the same numbers (run `make host-test`).

Usage:
    .venv/bin/python test_nn_parity.py            # checks ../weights.h
    .venv/bin/python test_nn_parity.py ../weights_working_w_pulse.h
"""

import sys
from pathlib import Path

import numpy as np

from fw_common import parse_weights_h, load_sb3_weights, nn_forward_np

HERE = Path(__file__).parent
DEFAULT_HEADER = HERE.parent / "weights.h"
GOLDEN_OUT = HERE.parent / "host_test" / "golden_vectors.h"

MODEL_CANDIDATES = [
    HERE / "runs/quad_hover/best_model.zip",
    HERE / "runs/quad_hover/final_model.zip",
    HERE / "runs/quad_hover_working_w_pulse/best_model.zip",
    HERE / "runs/quad_hover_working_w_pulse/final_model.zip",
]

PASS = "  \033[32m✓ PASS\033[0m"
FAIL = "  \033[31m✗ FAIL\033[0m"
WARN = "  \033[33m! WARN\033[0m"


def golden_observations() -> np.ndarray:
    """Deterministic obs set covering the flight envelope."""
    rows = []
    hover = np.array([0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0], np.float32)
    rows.append(("hover", hover))

    def mod(name, idx, val):
        o = hover.copy(); o[idx] = val; rows.append((name, o))

    mod("roll_err +10deg",  6,  10 / 45)
    mod("roll_err -10deg",  6, -10 / 45)
    mod("pitch_err +10deg", 7,  10 / 45)
    mod("pitch_err -10deg", 7, -10 / 45)
    mod("gyro_x +1rad/s",   0,  1 / 34.907)
    mod("gyro_z +2rad/s",   2,  2 / 34.907)
    mod("alt low (clamped)",  8, -0.05)
    mod("alt high (clamped)", 8, +0.05)
    mod("descending 1m/s",  9, -0.2)

    rng = np.random.default_rng(42)
    for i in range(5):
        rows.append((f"random_{i}",
                     rng.uniform(-1, 1, 12).astype(np.float32)))
    rows.append(("all_zero", np.zeros(12, np.float32)))
    rows.append(("saturated", np.tile([1.0, -1.0], 6).astype(np.float32)))
    return rows


def main():
    header = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HEADER
    print(f"═══ LAYER 0: NN parity — {header} ═══\n")
    failures = 0

    # ── 1. Parse ─────────────────────────────────────────────────
    w = parse_weights_h(header)
    n_params = sum(int(np.prod(w[k].shape)) for k in
                   ("W1", "B1", "W2", "B2", "W3", "B3"))
    ok = n_params == 1604
    print(f"{PASS if ok else FAIL}  weights.h parsed: {n_params} params "
          f"(expected 1604)")
    failures += not ok

    # ── 2. Provenance: which model does flash actually contain? ──
    print("\nMatching against trained models:")
    best_name, best_diff = None, np.inf
    for mp in MODEL_CANDIDATES:
        if not mp.exists():
            print(f"   (missing) {mp.relative_to(HERE)}")
            continue
        try:
            mw = load_sb3_weights(mp)
        except Exception as e:
            print(f"   (unreadable: {e}) {mp.relative_to(HERE)}")
            continue
        diff = max(float(np.abs(w[k] - mw[k]).max())
                   for k in ("W1", "B1", "W2", "B2", "W3", "B3"))
        tag = "  ← MATCH" if diff < 1e-6 else ""
        print(f"   max|Δ| = {diff:12.3e}   {mp.relative_to(HERE)}{tag}")
        if diff < best_diff:
            best_diff, best_name = diff, mp.relative_to(HERE)

    if best_diff < 1e-6:
        print(f"{PASS}  {header.name} was exported from: {best_name}")
    else:
        print(f"{FAIL}  {header.name} matches NO model zip "
              f"(closest: {best_name}, Δ={best_diff:.3e}).")
        print("        The weights in flash are NOT from any of these "
              "training runs — re-export before trusting anything.")
        failures += 1

    # ── 3. Hover throttle consistency ────────────────────────────
    from drone_env import HOVER_THROTTLE as ENV_HOVER, ACTION_SCALE as ENV_SCALE
    ht = w["NN_HOVER_THROTTLE"]
    ok = ht is not None and abs(ht - ENV_HOVER) < 1e-3
    print(f"\n{PASS if ok else FAIL}  NN_HOVER_THROTTLE {ht} vs training "
          f"{ENV_HOVER:.4f}")
    failures += not ok
    ok = w["NN_ACTION_SCALE"] == ENV_SCALE
    print(f"{PASS if ok else FAIL}  NN_ACTION_SCALE   {w['NN_ACTION_SCALE']} "
          f"vs training {ENV_SCALE}")
    failures += not ok

    # ── 4. Behavioural sanity on golden vectors ──────────────────
    print("\nGolden-vector outputs (from weights.h — i.e. what flash computes):")
    rows = golden_observations()
    outs = []
    for name, obs in rows:
        out = nn_forward_np(w, obs)
        outs.append(out)
        print(f"   {name:<20s} out = [{out[0]:+7.3f} {out[1]:+7.3f} "
              f"{out[2]:+7.3f} {out[3]:+7.3f}]")

    hover_out = outs[0]
    mag = float(np.abs(hover_out).max())
    if mag < 0.25:
        print(f"{PASS}  hover residual |max| = {mag:.3f} (< 0.25)")
    else:
        print(f"{WARN}  hover residual |max| = {mag:.3f} — the NN pushes "
              f"{mag*0.4*1000:.0f} µs of asymmetric throttle at PERFECT hover.")
        print("        It will fight itself on the real drone.")

    # Attitude response sign check: +roll error (rolled right) should
    # lower right-side motors... in NN index space: out[FL]+out[RL] (left)
    # vs out[FR]+out[RR] (right).  Rolled right (+err) → need MORE thrust on
    # the RIGHT side to push back → right > left.
    roll_out = outs[1]  # +10° roll error
    right = roll_out[1] + roll_out[2]
    left  = roll_out[0] + roll_out[3]
    ok = right > left
    print(f"{PASS if ok else FAIL}  +roll error → right-side thrust "
          f"({right:+.3f}) > left-side ({left:+.3f})")
    failures += not ok

    pitch_out = outs[3]  # +10° pitch error
    # +pitch err meaning depends on convention; check antisymmetry instead:
    pitch_out_n = outs[4]
    ok = float(np.dot(pitch_out - hover_out, pitch_out_n - hover_out)) < 0
    print(f"{PASS if ok else FAIL}  ±pitch errors produce opposite responses")
    failures += not ok

    # ── 5. Emit golden_vectors.h for the host-compiled C test ────
    GOLDEN_OUT.parent.mkdir(exist_ok=True)
    with open(GOLDEN_OUT, "w") as f:
        f.write("/* Auto-generated by test_nn_parity.py — do not edit.\n")
        f.write(f" * Source header: {header.name}\n */\n")
        f.write(f"#define N_GOLDEN {len(rows)}\n")
        f.write("static const char *golden_names[N_GOLDEN] = {\n")
        for name, _ in rows:
            f.write(f'    "{name}",\n')
        f.write("};\n")
        f.write("static const float golden_obs[N_GOLDEN][12] = {\n")
        for _, obs in rows:
            f.write("    {" + ", ".join(f"{v:.9e}f" for v in obs) + "},\n")
        f.write("};\n")
        f.write("static const float golden_out[N_GOLDEN][4] = {\n")
        for out in outs:
            f.write("    {" + ", ".join(f"{v:.9e}f" for v in out) + "},\n")
        f.write("};\n")
    print(f"\nWrote {GOLDEN_OUT.relative_to(HERE.parent)} — now run: "
          f"make host-test")

    print(f"\n{'═'*60}\nLayer 0 result: "
          f"{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
