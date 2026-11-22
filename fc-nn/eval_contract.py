#!/usr/bin/env python3
"""
eval_contract.py — acceptance gates for a policy against the FIRMWARE contract.

Run it on a checkpoint (.zip) before exporting, or on a weights.h after
exporting (same float32 forward pass as nn.c either way):

    python eval_contract.py runs/contract_s1/best_model.zip --out runs/contract_s1/contract_eval.json
    python eval_contract.py ../weights.h
    python eval_contract.py <model> --quick          # smoke test, not a verdict

Gates (all must pass; "advisory" rows are reported only):
  G1 bench      level, still, on the ground — the harness TEST 7 input
                [0,0,0, 0,0,0.5, 0,0, -0.05,0, 0,0]:  action spread × 400 < 50 µs
  G2 contract   evaluation.CONTRACT_SCENARIOS in QuadrotorEnv(firmware_contract=True):
                hover30 attitude RMS < 3°, |mean roll|,|mean pitch| < 1° (no steady
                lean), |mean yaw rate| < 5°/s (no steady spin), altitude RMSE < 10 cm;
                take-off from the ground; ±20° roll/pitch
                recoveries; randomised plant; wind.  Thrust ±25 % rows are advisory.
                Horizontal drift is reported, not gated (no position sensing).
  G3 firmware   test_firmware_sim.py scenarios B, C, D (fw_common.FirmwareController
                + Madgwick + baro.c replica): every run survives 30 s.
                E (thrust −25 %), F (loop stalls), G (ground take-off) advisory.
Exit status 0 = all gates pass.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from drone_env import QuadrotorEnv, HOVER_THROTTLE, ACTION_SCALE
from evaluation import evaluate, summarize, CONTRACT_SCENARIOS
from fw_common import FW, parse_weights_h, load_sb3_weights, nn_forward_np

BENCH_SPREAD_LIMIT_US = 50.0
BENCH_OBS = np.array([0, 0, 0, 0, 0, 0.5, 0, 0, -0.05, 0, 0, 0], dtype=np.float32)

# Input sweep from the 2026-11-21 investigation (level hover observation).
SWEEP = [
    ("bench: obs8=-0.05 (clamped), hvel=0", -0.05, 0.0, 0.0),
    ("in flight at target: obs8=0, hvel=0", 0.0, 0.0, 0.0),
    ("legacy sim crutch: obs8=0, hvel=(0.04,-0.021)", 0.0, 0.04, -0.021),
    ("legacy sim crutch: obs8=+0.092, hvel=sim", 0.092, 0.04, -0.021),
]


def load_weights(path):
    p = Path(path)
    return parse_weights_h(p) if p.suffix == ".h" else load_sb3_weights(p)


def action(weights, obs):
    return np.clip(nn_forward_np(weights, obs), -1.0, 1.0)


def bench_report(weights):
    """Steady-state PWMs the harness TEST 7 would print for this input."""
    a = action(weights, BENCH_OBS)
    alt = 0.0                                   # on the table: baro alt ≈ 0
    adj = float(np.clip(FW["ALT_KP"] * (FW["TARGET_ALT"] - alt), -FW["ALT_ADJ_MAX"], FW["ALT_ADJ_MAX"]))
    pwm = 1000 + 1000 * np.clip(HOVER_THROTTLE + adj + ACTION_SCALE * a, 0, 1)
    spread = float((a.max() - a.min()) * ACTION_SCALE * 1000)
    rows = []
    for label, o8, h10, h11 in SWEEP:
        obs = BENCH_OBS.copy(); obs[8], obs[10], obs[11] = o8, h10, h11
        aa = action(weights, obs)
        rows.append(dict(label=label, action=aa.tolist(),
                         spread_us=float((aa.max() - aa.min()) * ACTION_SCALE * 1000)))
    return dict(action=a.tolist(), pwm_us=pwm.tolist(), spread_us=spread,
                collective_us=float(a.mean() * ACTION_SCALE * 1000),
                # pair differences (µs): what kind of imbalance the spread is
                roll_us=float(((a[0] + a[3]) - (a[1] + a[2])) / 2 * ACTION_SCALE * 1000),
                pitch_us=float(((a[0] + a[1]) - (a[2] + a[3])) / 2 * ACTION_SCALE * 1000),
                yaw_us=float(((a[1] + a[3]) - (a[0] + a[2])) / 2 * ACTION_SCALE * 1000),
                passed=spread < BENCH_SPREAD_LIMIT_US, sweep=rows)


def contract_suite(weights, seeds, output=None):
    results = evaluate(lambda obs: nn_forward_np(weights, obs), seeds=seeds,
                       scenarios=CONTRACT_SCENARIOS, output=output,
                       env_factory=lambda: QuadrotorEnv(firmware_contract=True))
    summary = summarize(results)
    gated = [r for r in results if not r.get("advisory")]
    return dict(passed=all(r["passed"] for r in gated), summary=summary,
                gated_passed=sum(r["passed"] for r in gated), gated_total=len(gated))


def firmware_ladder(weights, runs, duration):
    from test_firmware_sim import run_ladder
    res = run_ladder(weights, "BCDEFG", runs=runs, duration=duration)
    out = {}
    for name, rs in res.items():
        gate = name[0] in "BCD"
        out[name] = dict(
            gated=gate, survived=sum(r["survived"] for r in rs), runs=len(rs),
            **{k: float(np.mean([r[k] for r in rs])) for k in
               ("alive_s", "alt_rmse", "att_rms", "mean_roll", "mean_pitch",
                "nn_std", "nn_sat", "mot_mean_us", "max_drift_m")})
    passed = all(v["survived"] == v["runs"] for v in out.values() if v["gated"])
    return dict(passed=passed, scenarios=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="SB3 .zip checkpoint or a weights.h")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--seeds", type=int, default=5, help="episodes per contract scenario")
    ap.add_argument("--fw-runs", type=int, default=5, help="runs per firmware scenario")
    ap.add_argument("--fw-duration", type=float, default=30.0)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed / 1 run / 10 s — checks plumbing, not a verdict")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.fw_runs, args.fw_duration = 1, 1, 10.0

    weights = load_weights(args.model)
    print(f"═══ Firmware-contract gates — {args.model} ═══\n")

    g1 = bench_report(weights)
    print("G1 bench vector (harness TEST 7 input)")
    print("   action " + " ".join(f"{v:+.3f}" for v in g1["action"]) +
          f"   spread {g1['spread_us']:.0f} µs (limit {BENCH_SPREAD_LIMIT_US:.0f})"
          f"   collective {g1['collective_us']:+.0f} µs")
    print("   expected TEST 7 PWM  FL {:.0f}  FR {:.0f}  RR {:.0f}  RL {:.0f} µs".format(*g1["pwm_us"]))
    print(f"   pair differences: left−right {g1['roll_us']:+.0f} µs, front−rear {g1['pitch_us']:+.0f} µs, "
          f"CW−CCW (yaw) {g1['yaw_us']:+.0f} µs")
    for r in g1["sweep"]:
        print(f"   sweep  {r['label']:<48} spread {r['spread_us']:5.0f} µs")
    print(f"   → {'PASS' if g1['passed'] else 'FAIL'}\n")

    print("G2 contract scenarios (QuadrotorEnv firmware_contract=True)")
    contract_json = (Path(args.out).with_name(Path(args.out).stem + "_episodes.json")
                     if args.out else None)
    g2 = contract_suite(weights, range(100, 100 + args.seeds), contract_json)
    print(f"   gated episodes passed {g2['gated_passed']}/{g2['gated_total']}"
          f"   → {'PASS' if g2['passed'] else 'FAIL'}\n")

    print("G3 firmware-in-the-loop (test_firmware_sim.py B–D gated, E–G advisory)")
    g3 = firmware_ladder(weights, args.fw_runs, args.fw_duration)
    print(f"   → {'PASS' if g3['passed'] else 'FAIL'}\n")

    overall = g1["passed"] and g2["passed"] and g3["passed"]
    report = dict(model=str(Path(args.model).resolve()), quick=args.quick,
                  passed=overall, bench=g1, contract=g2, firmware=g3)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"report → {args.out}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}"
          + ("  (quick mode — not a deployment verdict)" if args.quick else ""))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
