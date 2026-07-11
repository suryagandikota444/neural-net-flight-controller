"""
fw_common.py — Shared utilities for the FC test suite.

Provides:
  * parse_weights_h()   — read W1/B1/W2/B2/W3/B3 back out of a weights.h
  * nn_forward_np()     — float32 forward pass identical to nn.c
  * Madgwick            — Python port of ahrs.c (beta=0.05, NWU, az=+1 at rest)
  * FirmwareController  — exact replica of main.c's obs → NN → IIR → mixer path
  * Firmware constants  — single source of truth for the test files

Every constant here mirrors a #define in main.c / baro.c / weights.h.
If you change the firmware, change it here or the tests lie to you.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Firmware constants (must mirror main.c / baro.c / weights.h)
# ---------------------------------------------------------------------------
FW = dict(
    MAX_GYRO_RADS   = 34.907,     # main.c
    ANGLE_OBS_SCALE = 1.0 / 45.0, # main.c
    ALT_SCALE       = 2.0,        # main.c
    VVEL_SCALE      = 5.0,        # main.c
    TARGET_ALT      = 0.3,        # main.c
    ALT_OBS_CLAMP   = 0.05,       # main.c
    ALT_KP          = 0.15,       # main.c
    ALT_KD          = 0.02,       # main.c  (current working tree)
    ALT_ADJ_MAX     = 0.10,       # main.c
    HOVER_THROTTLE  = 0.3226,     # weights.h NN_HOVER_THROTTLE
    ACTION_SCALE    = 0.4,        # weights.h NN_ACTION_SCALE
    # Adaptive IIR (main.c step G)
    ALPHA_SLOW      = 0.08,
    ALPHA_FAST      = 0.60,
    SNAP_THRESH     = 0.15,
    # baro.c
    VVEL_ALPHA      = 0.1,
    BARO_DECIMATION = 4,
    AHRS_BETA       = 0.05,       # ahrs.c
    LOOP_DT         = 0.01,
)


# ---------------------------------------------------------------------------
# weights.h parser
# ---------------------------------------------------------------------------
_SHAPES = {
    "W1": (32, 12), "B1": (32,),
    "W2": (32, 32), "B2": (32,),
    "W3": (4, 32),  "B3": (4,),
}


def parse_weights_h(path: str | Path):
    """Parse a weights.h file back into numpy float32 arrays.

    Returns dict {name: ndarray} for W1,B1,W2,B2,W3,B3 plus
    'NN_HOVER_THROTTLE' and 'NN_ACTION_SCALE' floats.
    """
    text = Path(path).read_text()
    out = {}

    for name, shape in _SHAPES.items():
        m = re.search(
            rf"const\s+float\s+{name}(?:\[\d+\])+\s*=\s*\{{(.*?)\}};",
            text, re.DOTALL,
        )
        if m is None:
            raise ValueError(f"{path}: array {name} not found")
        vals = [float(v.rstrip("fF"))
                for v in re.findall(r"[-+0-9.eE]+f?", m.group(1))
                if any(c.isdigit() for c in v)]
        n_expect = int(np.prod(shape))
        if len(vals) != n_expect:
            raise ValueError(
                f"{path}: {name} has {len(vals)} values, expected {n_expect}")
        out[name] = np.array(vals, dtype=np.float32).reshape(shape)

    for macro in ("NN_HOVER_THROTTLE", "NN_ACTION_SCALE"):
        m = re.search(rf"#define\s+{macro}\s+([-+0-9.eE]+)f?", text)
        out[macro] = float(m.group(1)) if m else None

    return out


def load_sb3_weights(model_path: str | Path):
    """Extract the same six arrays from an SB3 PPO .zip (like export_weights.py)."""
    from stable_baselines3 import PPO
    model = PPO.load(str(model_path), device="cpu")
    policy_net = model.policy.mlp_extractor.policy_net
    linear = [m for m in policy_net if hasattr(m, "weight")]
    l1, l2 = linear
    l3 = model.policy.action_net
    f32 = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return {
        "W1": f32(l1.weight), "B1": f32(l1.bias),
        "W2": f32(l2.weight), "B2": f32(l2.bias),
        "W3": f32(l3.weight), "B3": f32(l3.bias),
    }


# ---------------------------------------------------------------------------
# NN forward pass — float32, same accumulation order as nn.c
# ---------------------------------------------------------------------------
def nn_forward_np(w: dict, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    h1 = np.maximum(w["W1"] @ obs + w["B1"], np.float32(0))
    h2 = np.maximum(w["W2"] @ h1 + w["B2"], np.float32(0))
    return (w["W3"] @ h2 + w["B3"]).astype(np.float32)


# ---------------------------------------------------------------------------
# Madgwick AHRS — Python port of ahrs.c (IMU variant, NWU, az=+1g at rest)
# ---------------------------------------------------------------------------
class Madgwick:
    def __init__(self, beta: float = FW["AHRS_BETA"]):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gx, gy, gz, ax, ay, az, dt):
        w, x, y, z = self.q

        # Rate of change from gyro: 0.5 * q ⊗ [0, gx, gy, gz]
        qdw = 0.5 * (-x * gx - y * gy - z * gz)
        qdx = 0.5 * ( w * gx + y * gz - z * gy)
        qdy = 0.5 * ( w * gy - x * gz + z * gx)
        qdz = 0.5 * ( w * gz + x * gy - y * gx)

        an = math.sqrt(ax * ax + ay * ay + az * az)
        if an > 1e-9:
            ax, ay, az = ax / an, ay / an, az / an
            # Objective: rotate world gravity [0,0,1] into body, compare w/ accel
            f0 = 2.0 * (x * z - w * y) - ax
            f1 = 2.0 * (w * x + y * z) - ay
            f2 = 2.0 * (0.5 - x * x - y * y) - az
            s0 = -2.0 * y * f0 + 2.0 * x * f1
            s1 =  2.0 * z * f0 + 2.0 * w * f1 - 4.0 * x * f2
            s2 = -2.0 * w * f0 + 2.0 * z * f1 - 4.0 * y * f2
            s3 =  2.0 * x * f0 + 2.0 * y * f1
            sn = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if sn > 1e-9:
                qdw -= self.beta * s0 / sn
                qdx -= self.beta * s1 / sn
                qdy -= self.beta * s2 / sn
                qdz -= self.beta * s3 / sn

        w += qdw * dt; x += qdx * dt; y += qdy * dt; z += qdz * dt
        n = math.sqrt(w * w + x * x + y * y + z * z)
        self.q = np.array([w, x, y, z]) / n

    def roll_pitch_deg(self):
        w, x, y, z = self.q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
        pitch = math.asin(sp)
        return math.degrees(roll), math.degrees(pitch)


# ---------------------------------------------------------------------------
# FirmwareController — replicates main.c steps F..H exactly
# ---------------------------------------------------------------------------
class FirmwareController:
    """One instance = the state main.c keeps between 100 Hz ticks.

    call step(gyro_rads, accel_g, roll_deg, pitch_deg, baro_alt_m, baro_vvel)
    -> motor_cmd[4] in [0,1], plus diagnostics dict.
    """

    def __init__(self, weights: dict,
                 use_adaptive_iir: bool = True,
                 clamp_alt_obs: bool = True,
                 use_alt_pid: bool = True,
                 train_alpha: float = 0.3):
        self.w = weights
        self.nn_smoothed = np.zeros(4, dtype=np.float32)
        self.use_adaptive_iir = use_adaptive_iir
        self.clamp_alt_obs = clamp_alt_obs
        self.use_alt_pid = use_alt_pid
        self.train_alpha = train_alpha

    def build_obs(self, gyro, accel, roll_deg, pitch_deg, alt_m, vvel):
        c = FW
        obs = np.zeros(12, dtype=np.float32)
        obs[0:3] = np.clip(np.asarray(gyro) / c["MAX_GYRO_RADS"], -1, 1)
        obs[3:6] = np.clip(np.asarray(accel) / 2.0, -1, 1)
        obs[6] = np.clip(roll_deg  * c["ANGLE_OBS_SCALE"], -1, 1)
        obs[7] = np.clip(pitch_deg * c["ANGLE_OBS_SCALE"], -1, 1)
        alt_err = alt_m - c["TARGET_ALT"]
        if self.clamp_alt_obs:
            obs[8] = np.clip(alt_err / c["ALT_SCALE"],
                             -c["ALT_OBS_CLAMP"], c["ALT_OBS_CLAMP"])
        else:
            obs[8] = np.clip(alt_err / c["ALT_SCALE"], -1, 1)
        obs[9] = np.clip(vvel / c["VVEL_SCALE"], -1, 1)
        obs[10] = 0.0   # firmware has no optical flow
        obs[11] = 0.0
        return obs

    def step(self, gyro, accel, roll_deg, pitch_deg, alt_m, vvel):
        c = FW
        obs = self.build_obs(gyro, accel, roll_deg, pitch_deg, alt_m, vvel)

        nn_out = np.clip(nn_forward_np(self.w, obs), -1.0, 1.0)
        raw = nn_out.copy()

        if self.use_adaptive_iir:
            # main.c: adaptive two-alpha IIR
            diff = nn_out - self.nn_smoothed
            alpha = np.where(np.abs(diff) >= c["SNAP_THRESH"],
                             c["ALPHA_FAST"], c["ALPHA_SLOW"])
            self.nn_smoothed = self.nn_smoothed + alpha * diff
        else:
            # training filter: fixed EMA
            self.nn_smoothed = (self.train_alpha * nn_out
                                + (1 - self.train_alpha) * self.nn_smoothed)
        nn_out = self.nn_smoothed

        alt_adj = 0.0
        if self.use_alt_pid:
            alt_adj = float(np.clip(
                c["ALT_KP"] * (c["TARGET_ALT"] - alt_m) + c["ALT_KD"] * (-vvel),
                -c["ALT_ADJ_MAX"], c["ALT_ADJ_MAX"]))

        base = c["HOVER_THROTTLE"] + alt_adj
        motor = np.clip(base + nn_out * c["ACTION_SCALE"], 0.0, 1.0)
        return motor, dict(obs=obs, nn_raw=raw, nn_smoothed=nn_out.copy(),
                           alt_adj=alt_adj)


# ---------------------------------------------------------------------------
# Measured noise from a real flight log (for sim-to-real scenarios)
# ---------------------------------------------------------------------------
def measure_flight_noise(csv_path: str | Path):
    """High-frequency noise std from a TEL flight log.

    Uses first differences / sqrt(2) so slow real motion doesn't inflate the
    estimate. Returns dict with gyro_std (rad/s), accel_std (g) and the raw
    per-axis numbers.
    """
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        r = _csv.DictReader(f)
        for row in r:
            rows.append([float(row[c]) for c in
                         ("gx_rads", "gy_rads", "gz_rads",
                          "ax_g", "ay_g", "az_g")])
    d = np.array(rows)
    diff_std = np.diff(d, axis=0).std(axis=0) / math.sqrt(2.0)
    return dict(
        gyro_std=float(diff_std[:3].mean()),
        accel_std=float(diff_std[3:].mean()),
        per_axis=diff_std,
    )
