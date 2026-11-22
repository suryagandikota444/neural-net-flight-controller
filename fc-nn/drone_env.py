"""
QuadrotorEnv – MuJoCo-based quadrotor hover environment.

Uses the MJCF model in quadrotor.xml (proper X-frame geometry with
visual props, motors, LEDs, ground plane, and target ghost).

Observation (12 values, same order as STM32 input vector):
  [0] gyro_x  / MAX_GYRO          -> [-1, 1]   MAX_GYRO = 2000 deg/s in rad/s = 34.91
  [1] gyro_y  / MAX_GYRO
  [2] gyro_z  / MAX_GYRO
  [3] accel_x / 2g                -> [-1, 1]   2g = normalization range (not chip FS)
  [4] accel_y / 2g
  [5] accel_z / 2g
  [6] roll_error  / 45.0          -> [-1, 1]   (relative to roll setpoint; ±45° maps to ±1)
  [7] pitch_error / 45.0                        (relative to pitch setpoint)
  [8] altitude_error / ALT_SCALE  -> [-1, 1]  (positive = above target)
  [9] vertical_vel / VVEL_SCALE   -> [-1, 1]  (positive = going up)
  [10] hvel_x / HVEL_SCALE        -> [-1, 1]  lateral velocity X (world frame)
  [11] hvel_y / HVEL_SCALE        -> [-1, 1]  lateral velocity Y (world frame)

  firmware_contract=True (default) — what main.c actually sends:
  [6..7] absolute roll / pitch (no setpoint)
  [8]  clamp((baro_alt - 0.3) / 2, ±0.05), baro_alt above the ground reference
  [9]  clamp(baro_vvel / 5, ±1), baro.c: 25 Hz difference + EMA(0.1)
  [10..11] = 0 (no GPS / optical flow)
  and the motor command gets main.c's P+D altitude term and arming ramp.
  firmware_contract=False reproduces the legacy (pre-2026-11-21) env exactly.

STM32 raw → obs conversion (ICM-42688-P, configured ±2000 dps / ±16g):
  obs[0..2] = raw_gyro  / GYRO_LSB_NORM   (= raw / 32768)
  obs[3..5] = raw_accel / ACCEL_LSB_NORM  (= raw / 4096)

  Derivation:
    gyro:  raw / 16.4 LSB_per_dps * pi/180 rad/deg
           -----------------------------------------  =  raw / 32768
                2000 deg/s * pi/180 (MAX_GYRO)

    accel: raw / 2048 LSB_per_g
           -------------------- = raw / 4096
                  2.0 g

  Cross-check with a flat board at rest (gravity ≈ 1g on AZ):
    AZ ≈ 2048 LSB  →  2048/4096 = 0.5  ✓  (matches MuJoCo at rest: 9.81/9.81/2 = 0.5)
    AX, AY ≈ 0 LSB →  0/4096   = 0.0  ✓

Action (4 values):
  Residual motor commands in [-1, 1]  (centered on hover throttle)
  Motor output = clip(HOVER_THROTTLE + action * ACTION_SCALE, 0, 1)
  Network output of 0 = hover.  Positive = more thrust, negative = less.
  STM32 firmware: motor_pwm_us = clip(HOVER_THR + nn_out * ACT_SCALE, 0, 1) * 1000 + 1000
  Motor layout – standard X-frame:
    0: Front-Left  (CCW)
    1: Front-Right (CW)
    2: Rear-Right  (CCW)
    3: Rear-Left   (CW)
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Physical / normalisation constants
# ---------------------------------------------------------------------------
MAX_GYRO       = 2000.0 * math.pi / 180.0   # 2000 deg/s  -> rad/s (≈34.9)
GRAVITY        = 9.81
DT             = 0.00025                     # 4 kHz  (250 µs)
FRAMESKIP      = 40                          # policy at 100 Hz (40 × 0.25 ms = 10 ms)
MAX_EP_STEPS   = 1000                        # 1000 policy steps × 10 ms = 10 s

# Nominal hover thrust per motor:  m*g / 4 = 1.118*9.81/4 ≈ 2.74 N
# (Measured 2026-11-21: 1.110 kg with battery — within 1 %, kept.)
BASE_MASS      = 1.118                       # kg  (F450 frame, 1400KV motors, 8045 props)
ARM            = 0.225                       # 225 mm arm  (F450: 450mm wheelbase / 2)

# --- Thrust map: MEASURED 2026-11-21 (firmware_contract mode) ---------------
# Kitchen-scale ramp, harness TEST 16, all four motors, same PWM:
#   experiments/bench-2026-11-21/THRUST_RAMP.md  (+ thrust_ramp_runs.csv)
# Fit over 1025–1275 µs (residuals ≤ 8 g):
#   total_g = 5843 · (u + 0.015)^1.745,   u = (PWM − 1000)/1000
# Per motor:  T(u) = THRUST_K · (u + THRUST_U0)^THRUST_EXP   [N]
# Real hover with the 1.110 kg drone ≈ 1372 µs. The old model was LINEAR
# (T = 8.5 N · u, hover 1323 µs): 49 µs too low, and around hover the real
# motors give ~1.45× more thrust per unit command than the linear model.
# Measured only up to u = 0.275 (≈ 60 % of hover thrust); above that it is
# an extrapolation of the fit.
THRUST_K       = 14.33                       # N per motor
THRUST_U0      = 0.015
THRUST_EXP     = 1.745
MAX_THRUST     = THRUST_K * (1.0 + THRUST_U0) ** THRUST_EXP   # ≈ 14.7 N at u = 1


def thrust_fraction(cmd, exp: float = THRUST_EXP, u0: float = THRUST_U0):
    """Motor command [0,1] → fraction of MAX_THRUST (MuJoCo actuator ctrl).
    exp = 1, u0 = 0 is the legacy linear map (returns cmd unchanged)."""
    cmd = np.asarray(cmd, dtype=np.float64)
    if exp == 1.0 and u0 == 0.0:
        return cmd
    return np.clip(cmd + u0, 0.0, None) ** exp / (1.0 + u0) ** exp


def hover_command(mass: float = BASE_MASS, max_thrust: float = MAX_THRUST,
                  exp: float = THRUST_EXP, u0: float = THRUST_U0) -> float:
    """Motor command at which four motors carry mass·g (inverse thrust map)."""
    frac = mass * GRAVITY / (4.0 * max_thrust)
    return float(frac ** (1.0 / exp) * (1.0 + u0) - u0)


# Residual action mapping: network outputs [-1,1], mapped to motor [0,1].
# HOVER_THROTTLE is the firmware base throttle (weights.h NN_HOVER_THROTTLE,
# written by export_weights.py with 6 decimals) — rounded to 6 dp here so
# the C float and this value describe the same equilibrium.
HOVER_THROTTLE = round(hover_command(), 6)   # 0.372638 → 1373 µs
ACTION_SCALE   = 0.4        # ±0.4 around hover → motor range [0, 0.77]

# --- Actuator delay: MEASURED 2026-11-22 (firmware_contract mode) ----------
# Flight-recorder fit of angular acceleration vs motor command
# (issues_solved/002_attitude_oscillation_2p7hz_actuator_delay): the real
# motors respond ~40 ms after the command (R² 0.78–0.82 vs 0.45–0.48 for an
# instant response). The ESC signal is 400 Hz, so this is ESC + rotor
# spin-up. The old simulator had NO delay; with it, the trained policy
# oscillated at 2.7 Hz on the real drone.
# Model: dead time of whole control ticks, then a first-order lag on thrust.
ACT_DELAY_TICKS_NOMINAL = 3        # 30 ms
ACT_TAU_NOMINAL         = 0.010    # 10 ms  → ≈ 40 ms effective
INERTIA_SCALE_NOMINAL   = 1.0


class ActuatorModel:
    """Dead time (whole policy ticks) + first-order lag on the thrust fraction.

    The lag is integrated exactly over each tick and the tick-AVERAGE thrust
    is applied (piecewise-constant ctrl over the physics substeps), so the
    cost is one numpy op per policy step.  delay 0 and tau 0 = instant."""

    def __init__(self, delay_ticks: int, tau_s: float, init_frac,
                 tick_s: float = FRAMESKIP * DT):
        self.delay = int(delay_ticks)
        self.tau = float(tau_s)
        self.tick = tick_s
        self.state = np.array(init_frac, dtype=np.float64)
        self.queue = [self.state.copy() for _ in range(self.delay)]

    def step(self, frac_cmd) -> np.ndarray:
        """Advance one tick with the new command; returns the tick-average thrust fraction."""
        frac_cmd = np.asarray(frac_cmd, dtype=np.float64)
        if self.delay:
            self.queue.append(frac_cmd.copy())
            target = self.queue.pop(0)
        else:
            target = frac_cmd
        if self.tau <= 0.0:
            self.state = target.copy()
            return self.state.copy()
        e = math.exp(-self.tick / self.tau)
        f0 = self.state
        avg = target + (f0 - target) * (self.tau / self.tick) * (1.0 - e)
        self.state = target + (f0 - target) * e
        return avg


# Legacy (firmware_contract=False) model — kept bit-identical for the
# regression fixtures: linear thrust, 8.5 N per motor.
LEGACY_MAX_THRUST     = 8.5
LEGACY_HOVER_THROTTLE = BASE_MASS * GRAVITY / (4 * LEGACY_MAX_THRUST)   # ≈ 0.323

# Motor spin directions: CCW = -1, CW = +1  (yaw torque sign)
MOTOR_DIRS     = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
KM_KF_RATIO    = 0.02                       # torque-to-thrust ratio

# Observation scaling
ALT_SCALE       = 2.0        # ±2 m altitude error maps to ±1
VVEL_SCALE      = 5.0        # ±5 m/s vertical velocity maps to ±1
HVEL_SCALE      = 3.0        # ±3 m/s horizontal velocity maps to ±1
ANGLE_ERR_SCALE = 45.0       # ±45° attitude error maps to ±1 (STM32: err_deg / 45.0)

# Setpoint limits (used by viewer/controller; training always commands 0)
MAX_CMD_ANGLE   = 45.0       # degrees – max roll/pitch command (final target)
MAX_YAW_RATE_CMD = 180.0     # deg/s   – max yaw-rate command

# ---------------------------------------------------------------------------
# Curriculum schedule — all values lerp from START → END as progress goes 0→1
# ---------------------------------------------------------------------------
CURRIC_TILT_START      = 5.0     # initial tilt range (deg) at progress=0
CURRIC_TILT_END        = 35.0    # final tilt range (deg) at progress=1
CURRIC_CMD_ANGLE_START = 15.0    # max setpoint cmd angle (deg) at progress=0
CURRIC_CMD_ANGLE_END   = 45.0    # max setpoint cmd angle (deg) at progress=1
CURRIC_HOVER_FRAC_START = 0.70   # fraction of episodes that are pure hover at start
CURRIC_HOVER_FRAC_END   = 0.50   # fraction at end (original 50/50 split)
CURRIC_SMOOTH_PEN_START = 0.15   # r_smooth penalty weight at start (gentler)
CURRIC_SMOOTH_PEN_END   = 0.30   # r_smooth penalty weight at end (full strength)

# STM32 raw-to-obs scale factors  (ICM-42688-P at ±2000 dps / ±16 g)
# Use these in the firmware when feeding the NN inference:
#   obs[0..2] = (float)raw_gyro  / GYRO_LSB_NORM
#   obs[3..5] = (float)raw_accel / ACCEL_LSB_NORM
GYRO_LSB_NORM   = 32768.0    # 16.4 LSB/(°/s) × 2000 °/s = 32800 ≈ 32768 (int16 full-scale)
ACCEL_LSB_NORM  = 4096.0     # 2048 LSB/g × 2g normalisation range

# Reward constants  (continuous + shaped)
CRASH_ALT       = 0.05
TILT_LIMIT_RAD  = math.radians(60.0)
CRASH_PENALTY   = 50.0

# Sim-to-real noise (added on top of MuJoCo sensor noise)
# Bumped up from 0.01/0.05 to better match real ICM-42688-P noise floor
GYRO_NOISE_STD  = 0.09      # rad/s  (was 0.01)
ACCEL_NOISE_STD = 0.22      # g      (was 0.05)

# AHRS latency simulation: std-dev of Gaussian noise added to attitude obs
# to simulate Madgwick filter lag (~100 ms convergence on hardware)
AHRS_LAG_STD    = 0.03      # ~1.5° of attitude jitter in normalised units

# Action smoothing alpha applied during training (must match ACTION_SMOOTH_ALPHA
# in main.c so the NN learns to expect filtered outputs)
TRAIN_ACTION_ALPHA = 0.3    # EMA: 0=frozen, 1=raw; same default as firmware

TARGET_ALT      = 1.0        # hover altitude (m) — LEGACY mode only

XML_PATH = str(pathlib.Path(__file__).parent / "quadrotor.xml")

# ---------------------------------------------------------------------------
# Firmware contract — what the real drone can sense and how main.c uses it.
#
# Source of truth: ../fc_config.h, ../main.c (steps F–H2), ../baro.c.
# test_contract.py parses fc_config.h / baro.c / main.c and asserts these
# values match, so a firmware change that is not mirrored here fails a test.
# ---------------------------------------------------------------------------
FW_TARGET_ALT      = 0.3     # fc_config.h TARGET_ALT  (m above baro ground ref)
FW_ALT_OBS_CLAMP   = 0.05    # fc_config.h ALT_OBS_CLAMP
FW_ALT_KP          = 0.15    # fc_config.h ALT_KP   (throttle per m)
FW_ALT_KD          = 0.02    # fc_config.h ALT_KD   (throttle per m/s)
FW_ALT_ADJ_MAX     = 0.10    # fc_config.h ALT_ADJ_MAX
FW_BARO_DECIMATION = 4       # baro.c BARO_DECIMATION (100 Hz / 4 = 25 Hz)
FW_VVEL_ALPHA      = 0.1     # baro.c VVEL_ALPHA
FW_RAMP_TICKS      = 50      # main.c RAMP_TICKS
FW_RAMP_FLOOR      = 0.06    # main.c H2 ramp min-spin floor (× ramp_frac)
FW_TILT_LIMIT_DEG  = 60.0    # main.c TILT_LIMIT_DEG (flight_safety motor cut)

# Height of the drone body origin when it rests on the floor in quadrotor.xml.
# This is where the barometer's power-on ground reference is taken, so
# baro altitude = z - GROUND_REST_Z.  (Measured: drop test settles at
# 0.0079636 m; test_contract.py re-measures it.)
GROUND_REST_Z      = 0.0079636
LIFTOFF_AGL        = 0.08    # m AGL: above this a ground start counts as airborne

# Bench measurement (experiments/bench-2026-11-21/screenlog.0, TEST 13, board
# flat): baro alt std 1.4 cm (peaks ±3.9 cm), vvel std 0.036 m/s.
BARO_NOISE_NOMINAL = 0.015

# Defaults for firmware_contract=True.  Pass a dict to QuadrotorEnv to override
# individual entries; pass False to get the legacy (pre-2026-11-21) env.
FIRMWARE_CONTRACT_DEFAULTS = dict(
    # REQUIRED 1 — no horizontal-velocity sensor: obs[10] = obs[11] = 0.
    zero_hvel_obs=True,
    # REQUIRED 2 — firmware altitude handling.
    target_alt_agl=FW_TARGET_ALT,
    alt_obs_clamp=FW_ALT_OBS_CLAMP,
    alt_kp=FW_ALT_KP,
    alt_kd=FW_ALT_KD,
    alt_adj_max=FW_ALT_ADJ_MAX,
    # RECOMMENDED 3 — barometer model for obs[8]/obs[9] and the P+D loop.
    baro_model=True,
    baro_decimation=FW_BARO_DECIMATION,
    baro_vvel_alpha=FW_VVEL_ALPHA,
    baro_noise_std=(0.01, 0.03),          # per-episode σ (m); nominal 0.015
    # RECOMMENDED 4 — thrust-map uncertainty.  max_thrust per motor =
    # MAX_THRUST × collective × per-motor, collective ~ U(range), per-motor
    # ~ U(1 ± motor_mismatch).  Legacy was per-motor U(0.9, 1.1) only.
    thrust_scale_range=(0.75, 1.25),
    motor_mismatch=0.10,
    # Measured thrust curve (see THRUST_K above). exp=1, u0=0 → linear.
    thrust_exp=THRUST_EXP,
    thrust_u0=THRUST_U0,
    # Measured actuator delay (see ActuatorModel): per episode, dead time
    # uniform over whole ticks in [lo, hi] and lag tau uniform in [lo, hi] s.
    # Nominal (domain_randomization=False): 3 ticks + 10 ms ≈ 40 ms.
    # Options "actuator_delay_ticks" / "motor_tau_s" at reset override.
    actuator_delay_ticks=(2, 5),
    motor_tau_s=(0.005, 0.03),
    # Phase the delay in with the training curriculum: both ranges are scaled
    # by curriculum progress (0 → instant motors, 1 → full measured range).
    # Why: with the full delay from step 0 the untrained policy crashes too
    # often for the curriculum to ever advance (runs delay_s1/s2_20261122
    # stalled at reward ~60, progress 0). Eval envs set this False so they
    # always see the full delay.
    actuator_curriculum=True,
    # Rotational inertia multiplier (on top of the mass scaling). The flight
    # fit suggested the real drone turns ~2x less per µs than the sim, but it
    # was hand-held — randomise until measured (bifilar pendulum).
    # Option "inertia_scale" at reset overrides.
    inertia_scale_range=(0.8, 1.5),
    # OPTIONAL 5 — ground starts with the firmware's 50-tick throttle ramp.
    ground_start_frac=0.25,
    ramp_ticks=FW_RAMP_TICKS,
    ramp_floor=FW_RAMP_FLOOR,
    liftoff_timeout_s=3.0,
    # Air starts (disturbance recovery): initial height above ground.
    air_start_alt_agl=(0.25, 0.8),
    # Firmware cuts motors at 60° (legacy env terminated at 90°).
    tilt_limit_deg=FW_TILT_LIMIT_DEG,
    # Firmware feeds absolute roll/pitch (no setpoint); legacy training also
    # practised tracking random attitude commands.  Off by default.
    attitude_setpoints=False,
    # Setpoint range when attitude_setpoints is on (pilot right stick,
    # fc_config.h RC_MAX_ANGLE_DEG 8): max |angle| in degrees grows from [0]
    # at curriculum 0 to [1] at 1; changes are slew-limited like rc.c
    # (RC_ANGLE_RATE_DPS). Horizontal-speed penalty is off while a lean is
    # commanded (leaning on purpose makes the drone move).
    cmd_angle_deg=(5.0, 10.0),
    setpoint_slew_dps=30.0,
    # Steady yaw disturbance torque (issue 003: the real drone spun ~15°/s CW
    # with the CW pair already ~36 µs/motor high, ≈ 0.02–0.04 N·m — e.g. a
    # tilted motor, which the sim never had). |τ| ~ U(range) × curriculum
    # progress (if actuator_curriculum), random sign. Option
    # "yaw_disturbance_nm" overrides.
    yaw_disturbance_nm=(0.0, 0.04),
    # Global multiplier on KM_KF_RATIO (0.02 is a guess; 10" props are often
    # 0.012–0.016, i.e. weaker yaw authority). Option "km_scale" overrides.
    km_scale_range=(0.6, 1.2),
    # Rotor drag coefficient range, N per m/s (see rotor_drag_force).
    # Option "rotor_drag" overrides; deterministic resets use the nominal.
    rotor_drag_range=(0.1, 0.5),
    # Yaw-rate term inside the tracking reward (contract mode only):
    #   r_track = exp(-5·tilt² - 4·alt_err² - w·min(gz, cap)²),
    #   w = yaw_track_weight × curriculum progress (if yaw_weight_curriculum)
    # Why: MuJoCo has no yaw drag and the legacy 0.06·gz² term is ~0.01/step
    # at 25°/s, so the first contract runs settled into a steady 20–30°/s spin.
    # At full weight 3: 25°/s costs 0.44/step, 10°/s 0.08, 5°/s 0.02.
    # Why capped and ramped: on 2026-11-21 an uncapped term from step 0
    # (first 0.5·|gz| linear, then 3·gz² inside the exp) made early, fast-
    # spinning policies earn ~nothing per step, so crashing on purpose paid
    # off — both runs collapsed (eval reward ~50 vs ~1250, curriculum stuck
    # at 0). With the ramp, early training is identical to the run that
    # learned to hover; the cap keeps r_track ≥ exp(-3·0.36) ≈ 0.34×.
    yaw_track_weight=3.0,
    yaw_track_cap=0.6,          # rad/s (34°/s)
    yaw_weight_curriculum=True,
    yaw_rate_penalty=0.0,       # linear |gz| penalty — keep 0 (see above)
)


def resolve_firmware_contract(firmware_contract) -> Optional[dict]:
    """True → defaults, dict → defaults updated with overrides, False/None → legacy."""
    if firmware_contract is None or firmware_contract is False:
        return None
    cfg = dict(FIRMWARE_CONTRACT_DEFAULTS)
    if isinstance(firmware_contract, dict):
        unknown = set(firmware_contract) - set(cfg) - {"enabled"}
        if unknown:
            raise ValueError(f"unknown firmware_contract keys: {sorted(unknown)}")
        if firmware_contract.get("enabled", True) is False:
            return None
        cfg.update({k: v for k, v in firmware_contract.items() if k != "enabled"})
    elif firmware_contract is not True:
        raise TypeError("firmware_contract must be bool or dict")
    for k in ("baro_noise_std", "thrust_scale_range", "air_start_alt_agl",
              "motor_tau_s", "inertia_scale_range", "cmd_angle_deg",
              "yaw_disturbance_nm", "km_scale_range", "rotor_drag_range"):
        v = cfg[k]
        cfg[k] = (float(v), float(v)) if np.isscalar(v) else tuple(float(x) for x in v)
    return cfg


def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo quaternion [w, x, y, z] → intrinsic Euler (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    # Roll (X)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll  = math.atan2(sinr, cosr)
    # Pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw (Z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw  = math.atan2(siny, cosy)
    return roll, pitch, yaw


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q [w, x, y, z]."""
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, q)
    return out


# Rotor drag (issue 003): spinning props resist sideways motion with a force
# in the rotor plane ≈ −c·v (body x/y). MuJoCo has no air drag, so without
# this the accelerometer never feels the drone sliding. c in N per m/s;
# literature values for ~1 kg quads are ~0.2–0.5. Not yet measured here.
ROTOR_DRAG_NOMINAL = 0.3


def rotor_drag_force(quat: np.ndarray, vel_world: np.ndarray, coeff: float) -> np.ndarray:
    """World-frame drag force −coeff·(v projected on the body x/y plane)."""
    if coeff == 0.0:
        return np.zeros(3)
    q_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    v_body = _quat_rotate_vec(q_inv, np.asarray(vel_world, dtype=np.float64))
    f_body = np.array([-coeff * v_body[0], -coeff * v_body[1], 0.0])
    return _quat_rotate_vec(quat, f_body)


class QuadrotorEnv(gym.Env):
    """
    MuJoCo quadrotor hover environment.

    Parameters
    ----------
    render_mode : str | None
        ``"human"`` for interactive MuJoCo viewer window.
        ``"rgb_array"`` for off-screen rendering (returns images).
        ``None`` for headless training (fastest).
    xml_path : str | None
        Override path to MJCF file (defaults to quadrotor.xml next to this file).
    firmware_contract : bool | dict
        True (default): train/evaluate against what the firmware actually
        provides — obs[10:12] = 0, baro-modelled obs[8]/obs[9] with the fixed
        0.3 m target and ±0.05 clamp, main.c's P+D altitude adjustment on the
        base throttle, wider thrust randomisation, ground starts with the
        throttle ramp.  See FIRMWARE_CONTRACT_DEFAULTS.  A dict overrides
        individual entries.  False: the legacy environment, bit-identical to
        the one the pre-2026-11-21 checkpoints were trained in.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # ---- construction ----------------------------------------------------

    def __init__(self, render_mode: Optional[str] = None, xml_path: Optional[str] = None,
                 firmware_contract=True):
        super().__init__()
        self.render_mode = render_mode
        self.contract = resolve_firmware_contract(firmware_contract)

        # Load model & create data
        _path = xml_path or XML_PATH
        self._model = mujoco.MjModel.from_xml_path(_path)
        self._model.opt.timestep = DT
        self._data  = mujoco.MjData(self._model)

        # Cache IDs
        self._drone_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        self._accel_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "accel"
        )
        self._gyro_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro"
        )
        self._pos_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "drone_pos"
        )
        self._quat_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "drone_quat"
        )
        self._angvel_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "drone_angvel"
        )
        self._vel_sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "drone_vel"
        )

        # Actuator IDs (fl, fr, rr, rl – matches XML order)
        self._act_ids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ["motor_fl", "motor_fr", "motor_rr", "motor_rl"]
        ])

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Rendering (lazy init)
        self._viewer   = None
        self._renderer = None

        # Episode state
        self._step_count    = 0
        self._max_ep_steps  = MAX_EP_STEPS
        self._max_thrust    = None
        self.last_motor_cmd = np.zeros(4)   # firmware motor command of the last step
        self._act_delay = ACT_DELAY_TICKS_NOMINAL
        self._act_tau = ACT_TAU_NOMINAL
        self._inertia_scale = INERTIA_SCALE_NOMINAL
        self._actuator = None
        self._km_kf        = None
        self._wind_force   = np.zeros(3)
        self._prev_action   = np.zeros(4, dtype=np.float32)
        self._smooth_action = np.zeros(4, dtype=np.float32)  # EMA state

        # Pilot setpoints (modified at runtime by viewer / controller)
        # Angles in radians, yaw_rate in rad/s, alt in metres
        self._cmd_roll      = 0.0
        self._cmd_pitch     = 0.0
        self._cmd_yaw_rate  = 0.0
        self._cmd_alt       = TARGET_ALT
        self._cmd_roll_tgt  = 0.0            # slew targets (contract setpoints)
        self._cmd_pitch_tgt = 0.0
        self._yaw_dist      = 0.0            # contract: steady yaw torque, N·m
        self._km_scale      = 1.0
        self._rotor_drag    = 0.0

        # When True (training), each episode gets a fresh random setpoint so
        # the model learns to track arbitrary roll/pitch/alt commands.
        # Set to False in the viewer so the pilot controls setpoints manually.
        self._randomize_setpoints = True

        # Curriculum progress: 0.0 = easy (small tilts, gentle penalties),
        # 1.0 = full difficulty.  Updated externally via set_curriculum().
        self._curriculum = 0.0
        self._nominal_mass = float(self._model.body_mass[self._drone_body_id])
        self._nominal_inertia = self._model.body_inertia[self._drone_body_id].copy()
        self._sensor_noise_enabled = True
        self._episode_randomize_setpoints = True

        # Firmware-contract state (unused in legacy mode)
        if self.contract is not None:
            self._cmd_alt = GROUND_REST_Z + self.contract["target_alt_agl"]
        self._meas_alt     = 0.0      # baro altitude (m above ground ref)
        self._meas_vvel    = 0.0      # baro vertical speed (m/s)
        self._baro_tick    = 0
        self._baro_noise   = 0.0
        self._ramp_counter = 0
        self._airborne     = True
        self._ground_start = False
        self._alt_adj      = 0.0
        self._thrust_scale = 1.0

    # ---- Firmware contract helpers ----------------------------------------

    @property
    def firmware_contract(self) -> bool:
        return self.contract is not None

    @property
    def target_z(self) -> float:
        """World-frame z the reward/termination/evaluation treat as target."""
        return float(self._cmd_alt)

    def contract_config(self) -> dict:
        """JSON-serialisable description of the observation/control contract."""
        if self.contract is None:
            return {"firmware_contract": False}
        return {"firmware_contract": True, **{k: (list(v) if isinstance(v, tuple) else v)
                                             for k, v in self.contract.items()},
                "ground_rest_z": GROUND_REST_Z, "liftoff_agl": LIFTOFF_AGL,
                "hover_throttle": HOVER_THROTTLE, "action_scale": ACTION_SCALE,
                "max_thrust_n": MAX_THRUST,
                "action_alpha": TRAIN_ACTION_ALPHA}

    def thrust_fraction(self, motor_cmd):
        """Motor command → MuJoCo ctrl (fraction of max thrust) for this env's
        thrust map: measured curve in contract mode, linear in legacy mode."""
        c = self.contract
        if c is None:
            return np.asarray(motor_cmd, dtype=np.float64)
        return thrust_fraction(motor_cmd, c["thrust_exp"], c["thrust_u0"])

    def _true_alt_agl(self) -> tuple[float, float]:
        pos = self._sensor(self._pos_sensor_id, 3)
        vel = self._sensor(self._vel_sensor_id, 3)
        return float(pos[2] - GROUND_REST_Z), float(vel[2])

    def _reset_alt_sensor(self, rng) -> None:
        """Baro has been running since power-on: start settled at the true height."""
        c = self.contract
        z_agl, vz = self._true_alt_agl()
        if c["baro_model"]:
            self._meas_alt = z_agl + float(rng.normal(0.0, self._baro_noise))
            self._meas_vvel = vz
            # Phase of the 25 Hz baro read relative to the 100 Hz loop
            self._baro_tick = int(rng.integers(0, c["baro_decimation"]))
        else:
            self._meas_alt, self._meas_vvel = z_agl, vz

    def _update_alt_sensor(self) -> None:
        """One 100 Hz tick of baro.c baro_update(): read every Nth tick,
        altitude = raw (ref − p)/12.015, vvel = EMA(α) of the difference."""
        c = self.contract
        z_agl, vz = self._true_alt_agl()
        if not c["baro_model"]:
            self._meas_alt, self._meas_vvel = z_agl, vz
            return
        self._baro_tick += 1
        if self._baro_tick < c["baro_decimation"]:
            return
        self._baro_tick = 0
        prev = self._meas_alt
        self._meas_alt = z_agl + float(self.np_random.normal(0.0, self._baro_noise))
        v_raw = (self._meas_alt - prev) / (DT * FRAMESKIP * c["baro_decimation"])
        a = c["baro_vvel_alpha"]
        self._meas_vvel = a * v_raw + (1.0 - a) * self._meas_vvel

    # ---- Curriculum API (called by CurriculumCallback) -------------------

    def set_curriculum(self, progress: float) -> None:
        """Set curriculum progress in [0, 1].  0 = easiest, 1 = hardest."""
        self._curriculum = float(np.clip(progress, 0.0, 1.0))

    def get_curriculum(self) -> float:
        """Return current curriculum progress."""
        return self._curriculum

    # ---- Setpoint API (called by viewer / controller) -------------------

    def set_setpoint(self,
                     roll_deg: float  = 0.0,
                     pitch_deg: float = 0.0,
                     yaw_rate_dps: float = 0.0,
                     alt_m: float    = TARGET_ALT) -> None:
        """Update pilot setpoints.  Call before each env.step() in the viewer.

        Parameters
        ----------
        roll_deg      : desired roll  angle  (degrees, + = right)
        pitch_deg     : desired pitch angle  (degrees, + = nose up)
        yaw_rate_dps  : desired yaw rate     (deg/s,   + = right)
        alt_m         : desired altitude     (metres)
        """
        self._cmd_roll     = math.radians(roll_deg)
        self._cmd_pitch    = math.radians(pitch_deg)
        self._cmd_roll_tgt, self._cmd_pitch_tgt = self._cmd_roll, self._cmd_pitch
        self._cmd_yaw_rate = math.radians(yaw_rate_dps)
        if self.contract is None:
            self._cmd_alt  = float(alt_m)
        # Firmware contract: the altitude target is fixed (TARGET_ALT above the
        # baro ground reference); alt_m is ignored.

    def _contract_cmd_angle(self) -> float:
        lo, hi = self.contract["cmd_angle_deg"]
        return lo + (hi - lo) * self._curriculum

    # ---- Gymnasium API ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = dict(options or {})
        self._episode_randomize_setpoints = bool(options.get(
            "randomize_setpoints", self._randomize_setpoints))
        self._sensor_noise_enabled = bool(options.get("sensor_noise", True))
        self._max_ep_steps = int(options.get("max_steps", self._max_ep_steps))

        c = self.contract

        # --- Domain randomisation -----------------------------------------
        if c is None:
            self._max_thrust = LEGACY_MAX_THRUST * rng.uniform(0.9, 1.1, size=4)
        else:
            # Real hover PWM is unmeasured: randomise the collective thrust map
            # as well as the per-motor mismatch.
            self._thrust_scale = float(rng.uniform(*c["thrust_scale_range"]))
            mm = c["motor_mismatch"]
            self._max_thrust = (MAX_THRUST * self._thrust_scale
                                * rng.uniform(1.0 - mm, 1.0 + mm, size=4))
            p = self._curriculum if c["actuator_curriculum"] else 1.0
            lo, hi = c["actuator_delay_ticks"]
            lo_p, hi_p = int(round(lo * p)), int(round(hi * p))
            self._act_delay = int(rng.integers(lo_p, max(lo_p, hi_p) + 1))
            tlo, thi = c["motor_tau_s"]
            self._act_tau = float(rng.uniform(tlo * p, thi * p))
            self._inertia_scale = float(rng.uniform(*c["inertia_scale_range"]))
        self._km_kf      = KM_KF_RATIO * rng.uniform(0.9, 1.1, size=4)
        mass             = BASE_MASS * float(rng.uniform(0.95, 1.05))
        cog_offset       = rng.uniform(-0.002, 0.002, size=3)
        cog_offset[2]    = 0.0
        wind_dir         = rng.normal(size=3)
        wind_dir        /= (np.linalg.norm(wind_dir) + 1e-8)
        self._wind_force = wind_dir * float(rng.uniform(0.0, 0.5))
        if c is not None:
            p_dist = self._curriculum if c["actuator_curriculum"] else 1.0
            self._yaw_dist = (float(rng.uniform(*c["yaw_disturbance_nm"])) * p_dist
                              * (1.0 if rng.random() < 0.5 else -1.0))
            self._km_scale = float(rng.uniform(*c["km_scale_range"]))
            self._rotor_drag = float(rng.uniform(*c["rotor_drag_range"]))

        # Public deterministic-reset interface for validation and experiments.
        # Still consume the usual domain draws, preserving the noise sequence.
        if not options.get("domain_randomization", True):
            self._max_thrust = np.full(4, MAX_THRUST if c is not None else LEGACY_MAX_THRUST)
            self._km_kf = np.full(4, KM_KF_RATIO)
            mass = BASE_MASS
            cog_offset = np.zeros(3)
            self._wind_force = np.zeros(3)
            self._thrust_scale = 1.0
            if c is not None:
                self._act_delay = ACT_DELAY_TICKS_NOMINAL
                self._act_tau = ACT_TAU_NOMINAL
                self._inertia_scale = INERTIA_SCALE_NOMINAL
                self._yaw_dist = 0.0
                self._km_scale = 1.0
                self._rotor_drag = ROTOR_DRAG_NOMINAL
        if c is not None:
            self._act_delay = int(options.get("actuator_delay_ticks", self._act_delay))
            self._act_tau = float(options.get("motor_tau_s", self._act_tau))
            self._inertia_scale = float(options.get("inertia_scale", self._inertia_scale))
            if self._act_delay < 0 or self._act_tau < 0 or self._inertia_scale <= 0:
                raise ValueError("actuator delay/tau must be >= 0 and inertia_scale > 0")
            self._yaw_dist = float(options.get("yaw_disturbance_nm", self._yaw_dist))
            self._km_scale = float(options.get("km_scale", self._km_scale))
            self._rotor_drag = float(options.get("rotor_drag", self._rotor_drag))
            if not (np.isfinite(self._yaw_dist) and self._km_scale > 0 and self._rotor_drag >= 0):
                raise ValueError("yaw disturbance must be finite, km_scale > 0, rotor_drag >= 0")
            self._km_kf = self._km_kf * self._km_scale
        if c is not None and "thrust_scale" in options:
            # Deterministic collective thrust-map error (e.g. 0.75 = 25 % weak)
            s = float(options["thrust_scale"])
            if not np.isfinite(s) or s <= 0:
                raise ValueError("thrust_scale must be positive")
            self._max_thrust = self._max_thrust / self._thrust_scale * s
            self._thrust_scale = s
        mass = float(options.get("mass", mass))
        cog_offset = np.asarray(options.get("cog_offset", cog_offset), dtype=float)
        self._wind_force = np.asarray(options.get("wind_force", self._wind_force), dtype=float)
        if not np.isfinite(mass) or mass <= 0 or cog_offset.shape != (3,) or not np.all(np.isfinite(cog_offset)):
            raise ValueError("mass must be positive and CoG a finite three-vector")
        if self._wind_force.shape != (3,) or not np.all(np.isfinite(self._wind_force)) or self._max_ep_steps <= 0:
            raise ValueError("wind must be a finite three-vector and max_steps positive")

        # --- Reset MuJoCo ------------------------------------------------
        mujoco.mj_resetData(self._model, self._data)

        # Apply randomised mass & CoG
        self._model.body_mass[self._drone_body_id] = mass
        self._model.body_ipos[self._drone_body_id] = cog_offset
        # Nominal XML inertia describes the entire aircraft. Density variation
        # scales mass and inertia together, never cumulatively across resets.
        self._model.body_inertia[self._drone_body_id] = (self._nominal_inertia * (mass / self._nominal_mass)
                                                         * (self._inertia_scale if c is not None else 1.0))
        # Compiled same-frame optimization otherwise ignores nonzero body_ipos.
        self._model.body_sameframe[self._drone_body_id] = 0
        mujoco.mj_setConst(self._model, self._data)
        for i, aid in enumerate(self._act_ids):
            self._model.actuator_gear[aid, 2] = self._max_thrust[i]

        # Random initial tilt — range grows with curriculum progress
        _tilt_max = CURRIC_TILT_START + (CURRIC_TILT_END - CURRIC_TILT_START) * self._curriculum
        tilt_x = float(rng.uniform(-_tilt_max, _tilt_max)) * math.pi / 180
        tilt_y = float(rng.uniform(-_tilt_max, _tilt_max)) * math.pi / 180
        cr, sr = math.cos(tilt_x / 2), math.sin(tilt_x / 2)
        cp, sp = math.cos(tilt_y / 2), math.sin(tilt_y / 2)
        quat = np.array([
            cr * cp,       # w
            sr * cp,       # x
            cr * sp,       # y
            -sr * sp,      # z
        ])

        # --- Setpoint randomisation (training only) -----------------------
        if c is not None:
            # Firmware: fixed altitude target, no attitude setpoint.
            self._cmd_roll = 0.0
            self._cmd_pitch = 0.0
            self._cmd_alt = GROUND_REST_Z + c["target_alt_agl"]
            if self._episode_randomize_setpoints and c["attitude_setpoints"]:
                _cmd_angle = self._contract_cmd_angle()
                _hover_frac = CURRIC_HOVER_FRAC_START + (CURRIC_HOVER_FRAC_END - CURRIC_HOVER_FRAC_START) * self._curriculum
                if rng.random() >= _hover_frac:
                    self._cmd_roll  = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
                    self._cmd_pitch = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
            # Slew-limited targets (rc.c): mid-episode changes ramp at
            # setpoint_slew_dps; the episode's first setpoint applies at once.
            self._cmd_roll_tgt, self._cmd_pitch_tgt = self._cmd_roll, self._cmd_pitch
        elif self._episode_randomize_setpoints:
            _hover_frac = CURRIC_HOVER_FRAC_START + (CURRIC_HOVER_FRAC_END - CURRIC_HOVER_FRAC_START) * self._curriculum
            _cmd_angle  = CURRIC_CMD_ANGLE_START  + (CURRIC_CMD_ANGLE_END  - CURRIC_CMD_ANGLE_START)  * self._curriculum
            if rng.random() < _hover_frac:
                # Hover practice — zero setpoints
                self._cmd_roll  = 0.0
                self._cmd_pitch = 0.0
                self._cmd_alt   = TARGET_ALT
            else:
                # Random setpoint tracking (angle range grows with curriculum)
                self._cmd_roll  = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
                self._cmd_pitch = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
                self._cmd_alt   = float(rng.uniform(0.6, 1.6))

        self._cmd_yaw_rate = 0.0
        if "setpoint" in options:
            self.set_setpoint(**options["setpoint"])
        if "initial_rpy_deg" in options:
            rpy = np.radians(np.asarray(options["initial_rpy_deg"], dtype=float))
            if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
                raise ValueError("initial_rpy_deg must be a finite three-vector")
            quat = np.zeros(4)
            # Extrinsic XYZ agrees with _quat_to_euler's roll/pitch/yaw convention.
            mujoco.mju_euler2Quat(quat, rpy, "XYZ")

        if c is None:
            # Legacy: always start at the commanded altitude.
            init_z = self._cmd_alt
            self._ground_start = False
        else:
            # Firmware arms on the ground; some episodes start there and go
            # through main.c's throttle ramp.  The rest start in the air
            # (tilted, per curriculum) to practise disturbance recovery.
            randomized = options.get("domain_randomization", True)
            ground_draw = float(rng.random())
            air_draw = float(rng.uniform(*c["air_start_alt_agl"]))
            self._ground_start = bool(options.get(
                "ground_start", randomized and ground_draw < c["ground_start_frac"]))
            if self._ground_start:
                init_z = GROUND_REST_Z
                if "initial_rpy_deg" not in options:
                    quat = np.array([1.0, 0.0, 0.0, 0.0])   # resting level
            else:
                alt_agl = options.get("initial_alt_agl",
                                      air_draw if randomized else c["target_alt_agl"])
                init_z = GROUND_REST_Z + float(alt_agl)
            if randomized and self._sensor_noise_enabled:
                self._baro_noise = float(rng.uniform(*c["baro_noise_std"]))
            elif self._sensor_noise_enabled:
                self._baro_noise = BARO_NOISE_NOMINAL
            else:
                self._baro_noise = 0.0
            self._baro_noise = float(options.get("baro_noise_std", self._baro_noise))
            self._ramp_counter = 0 if self._ground_start else c["ramp_ticks"]
            self._airborne = not self._ground_start
            self._alt_adj = 0.0

        # Free-joint qpos: [x, y, z, qw, qx, qy, qz]
        self._data.qpos[0:3] = options.get("initial_position", [0.0, 0.0, init_z])
        self._data.qpos[3:7] = quat
        self._data.qvel[:] = options.get("initial_velocity", np.zeros(6))

        mujoco.mj_forward(self._model, self._data)

        self._step_count    = 0
        self._prev_action   = np.zeros(4, dtype=np.float32)
        self._smooth_action = np.zeros(4, dtype=np.float32)
        # (if False, setpoints keep whatever the viewer last set via set_setpoint())

        if c is not None:
            self._reset_alt_sensor(rng)
            # Motors start at rest for a ground start, at hover thrust in the air.
            init = 0.0 if self._ground_start else float(self.thrust_fraction(HOVER_THROTTLE))
            self._actuator = ActuatorModel(self._act_delay, self._act_tau, np.full(4, init))

        obs = self._get_obs()
        info = {"mass_kg": float(self._model.body_subtreemass[self._drone_body_id]),
                "cog_offset_m": cog_offset.copy()}
        if c is not None:
            info.update(ground_start=self._ground_start, thrust_scale=self._thrust_scale,
                        baro_noise_std=self._baro_noise, target_z=self.target_z,
                        actuator_delay_ticks=self._act_delay, motor_tau_s=self._act_tau,
                        inertia_scale=self._inertia_scale, yaw_disturbance_nm=self._yaw_dist,
                        km_scale=self._km_scale, rotor_drag=self._rotor_drag)
        return obs, info

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain four finite values")
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self._step_count += 1

        # --- Action EMA smoothing (matches ACTION_SMOOTH_ALPHA in firmware) ---
        # The NN trains with the same low-pass filter so it learns to expect
        # smoothed outputs, closing the sim-to-real gap.
        self._smooth_action = (
            TRAIN_ACTION_ALPHA * action
            + (1.0 - TRAIN_ACTION_ALPHA) * self._smooth_action
        ).astype(np.float32)
        smooth_action = self._smooth_action.astype(np.float64)

        # --- Residual → motor command mapping ----------------------------
        c = self.contract
        ramped = False
        if c is None:
            motor_cmd = np.clip(
                LEGACY_HOVER_THROTTLE + smooth_action * ACTION_SCALE, 0.0, 1.0
            )
        elif self._ramp_counter < c["ramp_ticks"]:
            # main.c H2: first RAMP_TICKS after arming scale base AND residual,
            # with a min-spin floor; the altitude P+D is not applied yet.
            self._ramp_counter += 1
            ramped = True
            frac = self._ramp_counter / c["ramp_ticks"]
            self._alt_adj = 0.0
            motor_cmd = np.clip(frac * (HOVER_THROTTLE + smooth_action * ACTION_SCALE),
                                frac * c["ramp_floor"], 1.0)
        else:
            # main.c F/H: P+D altitude hold on the base throttle, using the
            # same baro sample the NN saw in the observation it acted on.
            self._alt_adj = float(np.clip(
                c["alt_kp"] * (c["target_alt_agl"] - self._meas_alt)
                + c["alt_kd"] * (-self._meas_vvel),
                -c["alt_adj_max"], c["alt_adj_max"]))
            motor_cmd = np.clip(
                HOVER_THROTTLE + self._alt_adj + smooth_action * ACTION_SCALE, 0.0, 1.0
            )

        # --- Motor forces via actuators -----------------------------------
        # motor_cmd is what the firmware writes (PWM = 1000 + 1000·cmd);
        # ctrl is the fraction of max thrust that command produces.
        self.last_motor_cmd = motor_cmd.copy()
        frac = self.thrust_fraction(motor_cmd)
        if c is not None:
            # Real motors: ~40 ms dead time + spin-up lag (ActuatorModel).
            frac = self._actuator.step(frac)
        thrusts = frac * self._max_thrust
        for i, aid in enumerate(self._act_ids):
            self._model.actuator_gear[aid, 2] = self._max_thrust[i]
        self._data.ctrl[self._act_ids] = frac

        # --- Yaw reaction torque (body Z) → world frame -------------------
        yaw_torque = float(np.sum(MOTOR_DIRS * self._km_kf * thrusts))
        quat = self._data.qpos[3:7].copy()
        force_world = self._wind_force
        if c is not None:
            # Steady airframe yaw torque (tilted motor etc.) + rotor drag
            yaw_torque += self._yaw_dist
            force_world = force_world + rotor_drag_force(quat, self._data.qvel[0:3], self._rotor_drag)
        torque_world = _quat_rotate_vec(quat, np.array([0.0, 0.0, yaw_torque]))

        # External forces on the drone body: wind (+ drag) + yaw torque
        self._data.xfrc_applied[self._drone_body_id, 0:3] = force_world
        self._data.xfrc_applied[self._drone_body_id, 3:6] = torque_world

        # Step physics FRAMESKIP times (policy at 100 Hz, physics at 4 kHz)
        for _ in range(FRAMESKIP):
            mujoco.mj_step(self._model, self._data)

        # --- Mid-episode setpoint variation (training only) ---------------
        if c is not None:
            # Firmware has no altitude setpoint changes; attitude commands
            # only if explicitly enabled in the contract config.
            if (self._episode_randomize_setpoints and c["attitude_setpoints"]
                    and self._step_count % 200 == 0):
                rng = self.np_random
                _cmd_angle = self._contract_cmd_angle()
                r = rng.random()
                if r < 0.35:
                    self._cmd_roll_tgt = self._cmd_pitch_tgt = 0.0
                elif r < 0.7:
                    self._cmd_roll_tgt  = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
                    self._cmd_pitch_tgt = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
            if c["attitude_setpoints"]:
                step = math.radians(c["setpoint_slew_dps"]) * DT * FRAMESKIP
                self._cmd_roll += float(np.clip(self._cmd_roll_tgt - self._cmd_roll, -step, step))
                self._cmd_pitch += float(np.clip(self._cmd_pitch_tgt - self._cmd_pitch, -step, step))
            self._update_alt_sensor()
        elif self._episode_randomize_setpoints and self._step_count % 200 == 0:
            rng = self.np_random
            _cmd_angle = CURRIC_CMD_ANGLE_START + (CURRIC_CMD_ANGLE_END - CURRIC_CMD_ANGLE_START) * self._curriculum
            r = rng.random()
            if r < 0.35:
                # Return to hover (zero setpoint) — practice stabilisation
                self._cmd_roll  = 0.0
                self._cmd_pitch = 0.0
            elif r < 0.7:
                # New random attitude command (range grows with curriculum)
                self._cmd_roll  = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
                self._cmd_pitch = math.radians(float(rng.uniform(-_cmd_angle, _cmd_angle)))
            if rng.random() < 0.25:
                self._cmd_alt = float(rng.uniform(0.6, 1.6))

        # --- Observation & reward ----------------------------------------
        obs    = self._get_obs()
        reward = self._compute_reward(action)

        pos  = self._sensor(self._pos_sensor_id, 3)
        quat = self._sensor(self._quat_sensor_id, 4)
        roll, pitch, _ = _quat_to_euler(quat)

        no_liftoff = False
        if c is None:
            crashed   = pos[2] < CRASH_ALT
            flipped   = abs(roll) > math.pi / 2 or abs(pitch) > math.pi / 2
        else:
            # Ground starts sit below CRASH_ALT until they lift off, so ground
            # contact only counts as a crash once the drone has been airborne.
            if not self._airborne and pos[2] - GROUND_REST_Z > LIFTOFF_AGL:
                self._airborne = True
            crashed = self._airborne and pos[2] < CRASH_ALT
            no_liftoff = (not self._airborne and
                          self._step_count * DT * FRAMESKIP >= c["liftoff_timeout_s"])
            lim = math.radians(c["tilt_limit_deg"])
            flipped = abs(roll) > lim or abs(pitch) > lim
        too_far   = abs(pos[2] - self._cmd_alt) > 3.0   # >3m from cmd alt
        terminated = crashed or flipped or too_far or no_liftoff
        truncated  = self._step_count >= self._max_ep_steps

        if terminated:
            reward -= CRASH_PENALTY

        self._prev_action = action.astype(np.float32)  # store raw action for rate penalty
        reason = ("ground_contact" if crashed else "tilt_limit" if flipped
                  else "altitude_limit" if too_far else "no_liftoff" if no_liftoff else None)
        info = {"termination_reason": reason}
        if c is not None:
            info.update(alt_throttle_adj=self._alt_adj, baro_alt=self._meas_alt,
                        baro_vvel=self._meas_vvel, ramping=ramped)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_offscreen()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---- Sensor / state helpers ------------------------------------------

    def _sensor(self, sid: int, dim: int) -> np.ndarray:
        adr = self._model.sensor_adr[sid]
        return self._data.sensordata[adr : adr + dim].copy()

    def _get_obs(self) -> np.ndarray:
        # Gyro (body-frame, rad/s)
        gyro_raw = self._sensor(self._gyro_sensor_id, 3)
        gyro_raw += self.np_random.normal(0, GYRO_NOISE_STD, 3) * self._sensor_noise_enabled
        gyro_norm = np.clip(gyro_raw / MAX_GYRO, -1.0, 1.0).astype(np.float32)

        # Accel (body-frame, m/s² → g → normalise by 2g)
        accel_raw = self._sensor(self._accel_sensor_id, 3)
        accel_g   = accel_raw / GRAVITY
        accel_g  += self.np_random.normal(0, ACCEL_NOISE_STD, 3) * self._sensor_noise_enabled
        accel_norm = np.clip(accel_g / 2.0, -1.0, 1.0).astype(np.float32)

        # Attitude error (roll & pitch, relative to setpoint)
        # Add AHRS lag noise to simulate Madgwick filter latency on hardware
        quat = self._sensor(self._quat_sensor_id, 4)
        roll, pitch, _ = _quat_to_euler(quat)
        ahrs_noise = self.np_random.normal(0, AHRS_LAG_STD, 2) * self._sensor_noise_enabled
        roll_err  = np.float32(np.clip(math.degrees(roll  - self._cmd_roll)  / ANGLE_ERR_SCALE + ahrs_noise[0], -1.0, 1.0))
        pitch_err = np.float32(np.clip(math.degrees(pitch - self._cmd_pitch) / ANGLE_ERR_SCALE + ahrs_noise[1], -1.0, 1.0))

        c = self.contract
        pos = self._sensor(self._pos_sensor_id, 3)
        vel = self._sensor(self._vel_sensor_id, 3)
        if c is None:
            # Altitude error (positive = above target, relative to setpoint)
            alt_err = np.float32(np.clip((pos[2] - self._cmd_alt) / ALT_SCALE, -1.0, 1.0))
            # Vertical velocity
            vvel = np.float32(np.clip(vel[2] / VVEL_SCALE, -1.0, 1.0))
            # Horizontal velocity (world frame X/Y) – obs[10], obs[11]
            hvel_x = np.float32(np.clip(vel[0] / HVEL_SCALE, -1.0, 1.0))
            hvel_y = np.float32(np.clip(vel[1] / HVEL_SCALE, -1.0, 1.0))
        else:
            # main.c step F: baro altitude vs fixed TARGET_ALT, tightly clamped
            clamp = c["alt_obs_clamp"]
            alt_err = np.float32(np.clip((self._meas_alt - c["target_alt_agl"]) / ALT_SCALE,
                                         -clamp, clamp))
            vvel = np.float32(np.clip(self._meas_vvel / VVEL_SCALE, -1.0, 1.0))
            if c["zero_hvel_obs"]:
                # No GPS / optical flow on the drone: firmware sends 0.
                hvel_x = hvel_y = np.float32(0.0)
            else:
                hvel_x = np.float32(np.clip(vel[0] / HVEL_SCALE, -1.0, 1.0))
                hvel_y = np.float32(np.clip(vel[1] / HVEL_SCALE, -1.0, 1.0))

        return np.array([
            gyro_norm[0],  gyro_norm[1],  gyro_norm[2],
            accel_norm[0], accel_norm[1], accel_norm[2],
            roll_err, pitch_err,
            alt_err, vvel,
            hvel_x, hvel_y,
        ], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        pos     = self._sensor(self._pos_sensor_id, 3)
        quat    = self._sensor(self._quat_sensor_id, 4)
        vel     = self._sensor(self._vel_sensor_id, 3)
        ang_vel = self._sensor(self._angvel_sensor_id, 3)
        roll, pitch, _ = _quat_to_euler(quat)

        # Errors relative to pilot setpoints
        roll_err  = roll  - self._cmd_roll
        pitch_err = pitch - self._cmd_pitch
        alt_err   = pos[2] - self._cmd_alt

        # ---- Reward components ----

        # 1. Positive tracking reward (exponential — smooth gradient toward target)
        #    Always in [0, 1], peaks at 1.0 when attitude + altitude are perfect.
        #    This is the PRIMARY learning signal — gives the agent a clear
        #    direction to improve, even far from the target.
        orient_err_sq = roll_err ** 2 + pitch_err ** 2
        yaw_sq = 0.0
        if self.contract is not None:
            c = self.contract
            w = c["yaw_track_weight"] * (self._curriculum if c["yaw_weight_curriculum"] else 1.0)
            yaw_sq = w * min(abs(float(ang_vel[2])), c["yaw_track_cap"]) ** 2
        r_track = math.exp(-5.0 * orient_err_sq - 4.0 * alt_err ** 2 - yaw_sq)

        # 2. Alive bonus
        r_alive = 0.5

        # 3. Velocity damping (matches PPO_1 proven coefficients)
        hvel_sq = vel[0] ** 2 + vel[1] ** 2
        if self.contract is not None and (self._cmd_roll_tgt != 0.0 or self._cmd_pitch_tgt != 0.0):
            hvel_sq = 0.0        # a commanded lean is supposed to move the drone
        r_vel = -0.02 * hvel_sq - 0.15 * vel[2] ** 2

        # 4. Angular velocity penalty (slightly heavier on yaw)
        r_angvel = -0.02 * (ang_vel[0] ** 2 + ang_vel[1] ** 2) \
                 - 0.06 * ang_vel[2] ** 2

        # 5. Action smoothness: penalise motor command rate-of-change
        # Use raw action (not smoothed) so the gradient punishes the NN directly
        # Penalty ramps from gentle (0.15) to full (0.30) with curriculum
        _smooth_pen = CURRIC_SMOOTH_PEN_START + (CURRIC_SMOOTH_PEN_END - CURRIC_SMOOTH_PEN_START) * self._curriculum
        action_delta = action - self._prev_action
        r_smooth = -_smooth_pen * float(np.dot(action_delta, action_delta))

        # 6. Motor symmetry: light penalty on diagonal bias
        diag_diff = (action[0] + action[3]) - (action[1] + action[2])
        r_symm = -0.05 * diag_diff ** 2

        if self.contract is not None:
            r_angvel -= self.contract["yaw_rate_penalty"] * abs(float(ang_vel[2]))   # 0 by default

        return r_track + r_alive + r_vel + r_angvel + r_smooth + r_symm

    # ---- Rendering -------------------------------------------------------

    def _render_human(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._model, self._data,
                show_left_ui=False, show_right_ui=False,
            )
            self._viewer.cam.azimuth   = 135
            self._viewer.cam.elevation = -25
            self._viewer.cam.distance  = 2.0
        # Camera tracks the drone
        pos = self._data.qpos[0:3]
        self._viewer.cam.lookat[:] = pos
        self._viewer.sync()

    def _render_offscreen(self, width: int = 640, height: int = 480):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height, width)
        self._renderer.update_scene(self._data)
        return self._renderer.render()
