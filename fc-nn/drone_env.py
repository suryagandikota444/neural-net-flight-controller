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
  [6] roll_error  / 180.0         -> [-1, 1]   (relative to roll setpoint)
  [7] pitch_error / 180.0                       (relative to pitch setpoint)
  [8] altitude_error / ALT_SCALE  -> [-1, 1]  (positive = above target)
  [9] vertical_vel / VVEL_SCALE   -> [-1, 1]  (positive = going up)
  [10] hvel_x / HVEL_SCALE        -> [-1, 1]  lateral velocity X (world frame)
  [11] hvel_y / HVEL_SCALE        -> [-1, 1]  lateral velocity Y (world frame)

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
  Motor PWM commands in [0, 1]  (scaled to 1000–2000 µs at export)
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
MAX_EP_STEPS   = 500                         # 500 policy steps × 10 ms = 5 s

# Nominal hover thrust per motor:  m*g / 4 = 0.5*9.81/4 ≈ 1.23 N
BASE_MASS      = 0.5                         # kg
MAX_THRUST     = 3.0                         # N per motor (gives ~2.4:1 T/W)
ARM            = 0.1                         # 10 cm arm

# Motor spin directions: CCW = -1, CW = +1  (yaw torque sign)
MOTOR_DIRS     = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
KM_KF_RATIO    = 0.02                       # torque-to-thrust ratio

# Observation scaling
ALT_SCALE       = 2.0        # ±2 m altitude error maps to ±1
VVEL_SCALE      = 5.0        # ±5 m/s vertical velocity maps to ±1
HVEL_SCALE      = 3.0        # ±3 m/s horizontal velocity maps to ±1

# Setpoint limits (used by viewer/controller; training always commands 0)
MAX_CMD_ANGLE   = 30.0       # degrees – max roll/pitch command
MAX_YAW_RATE_CMD = 180.0     # deg/s   – max yaw-rate command

# STM32 raw-to-obs scale factors  (ICM-42688-P at ±2000 dps / ±16 g)
# Use these in the firmware when feeding the NN inference:
#   obs[0..2] = (float)raw_gyro  / GYRO_LSB_NORM
#   obs[3..5] = (float)raw_accel / ACCEL_LSB_NORM
GYRO_LSB_NORM   = 32768.0    # 16.4 LSB/(°/s) × 2000 °/s = 32800 ≈ 32768 (int16 full-scale)
ACCEL_LSB_NORM  = 4096.0     # 2048 LSB/g × 2g normalisation range

# Reward constants  (continuous + shaped)
CRASH_ALT       = 0.05
TILT_LIMIT_RAD  = math.radians(45.0)
CRASH_PENALTY   = 50.0

# Sim-to-real noise (added on top of MuJoCo sensor noise)
GYRO_NOISE_STD  = 0.01      # rad/s
ACCEL_NOISE_STD = 0.05      # g

TARGET_ALT      = 1.0        # hover altitude (m)

XML_PATH = str(pathlib.Path(__file__).parent / "quadrotor.xml")


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
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    # ---- construction ----------------------------------------------------

    def __init__(self, render_mode: Optional[str] = None, xml_path: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

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
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Rendering (lazy init)
        self._viewer   = None
        self._renderer = None

        # Episode state
        self._step_count    = 0
        self._max_ep_steps  = MAX_EP_STEPS
        self._max_thrust    = None
        self._km_kf        = None
        self._wind_force   = np.zeros(3)
        self._prev_action  = np.zeros(4, dtype=np.float32)

        # Pilot setpoints (modified at runtime by viewer / controller)
        # Angles in radians, yaw_rate in rad/s, alt in metres
        self._cmd_roll      = 0.0
        self._cmd_pitch     = 0.0
        self._cmd_yaw_rate  = 0.0
        self._cmd_alt       = TARGET_ALT

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
        self._cmd_yaw_rate = math.radians(yaw_rate_dps)
        self._cmd_alt      = float(alt_m)

    # ---- Gymnasium API ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # --- Domain randomisation -----------------------------------------
        self._max_thrust = MAX_THRUST * rng.uniform(0.9, 1.1, size=4)
        self._km_kf      = KM_KF_RATIO * rng.uniform(0.9, 1.1, size=4)
        mass             = BASE_MASS * float(rng.uniform(0.95, 1.05))
        cog_offset       = rng.uniform(-0.002, 0.002, size=3)
        cog_offset[2]    = 0.0
        wind_dir         = rng.normal(size=3)
        wind_dir        /= (np.linalg.norm(wind_dir) + 1e-8)
        self._wind_force = wind_dir * float(rng.uniform(0.0, 0.5))

        # --- Reset MuJoCo ------------------------------------------------
        mujoco.mj_resetData(self._model, self._data)

        # Apply randomised mass & CoG
        self._model.body_mass[self._drone_body_id] = mass
        self._model.body_ipos[self._drone_body_id] = cog_offset

        # Random initial tilt ±5 deg
        tilt_x = float(rng.uniform(-5, 5)) * math.pi / 180
        tilt_y = float(rng.uniform(-5, 5)) * math.pi / 180
        cr, sr = math.cos(tilt_x / 2), math.sin(tilt_x / 2)
        cp, sp = math.cos(tilt_y / 2), math.sin(tilt_y / 2)
        quat = np.array([
            cr * cp,       # w
            sr * cp,       # x
            cr * sp,       # y
            -sr * sp,      # z
        ])

        # Free-joint qpos: [x, y, z, qw, qx, qy, qz]
        self._data.qpos[0:3] = [0.0, 0.0, 1.0]
        self._data.qpos[3:7] = quat

        mujoco.mj_forward(self._model, self._data)

        self._step_count  = 0
        self._prev_action = np.zeros(4, dtype=np.float32)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0).astype(np.float64)
        self._step_count += 1

        # --- Motor forces via actuators -----------------------------------
        thrusts = action * self._max_thrust
        for i, aid in enumerate(self._act_ids):
            self._model.actuator_gear[aid, 2] = self._max_thrust[i]
        self._data.ctrl[self._act_ids] = action

        # --- Yaw reaction torque (body Z) → world frame -------------------
        yaw_torque = float(np.sum(MOTOR_DIRS * self._km_kf * thrusts))
        quat = self._data.qpos[3:7].copy()
        torque_world = _quat_rotate_vec(quat, np.array([0.0, 0.0, yaw_torque]))

        # External forces on the drone body: wind + yaw torque
        self._data.xfrc_applied[self._drone_body_id, 0:3] = self._wind_force
        self._data.xfrc_applied[self._drone_body_id, 3:6] = torque_world

        # Step physics FRAMESKIP times (policy at 100 Hz, physics at 4 kHz)
        for _ in range(FRAMESKIP):
            mujoco.mj_step(self._model, self._data)

        # --- Observation & reward ----------------------------------------
        obs    = self._get_obs()
        reward = self._compute_reward(action)

        pos  = self._sensor(self._pos_sensor_id, 3)
        quat = self._sensor(self._quat_sensor_id, 4)
        roll, pitch, _ = _quat_to_euler(quat)

        crashed   = pos[2] < CRASH_ALT
        flipped   = abs(roll) > math.pi / 2 or abs(pitch) > math.pi / 2
        too_far   = abs(pos[2] - TARGET_ALT) > 3.0   # >3m from target alt
        terminated = crashed or flipped or too_far
        truncated  = self._step_count >= self._max_ep_steps

        if terminated:
            reward -= CRASH_PENALTY

        self._prev_action = action.astype(np.float32)
        return obs, float(reward), terminated, truncated, {}

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
        gyro_raw += self.np_random.normal(0, GYRO_NOISE_STD, 3)
        gyro_norm = np.clip(gyro_raw / MAX_GYRO, -1.0, 1.0).astype(np.float32)

        # Accel (body-frame, m/s² → g → normalise by 2g)
        accel_raw = self._sensor(self._accel_sensor_id, 3)
        accel_g   = accel_raw / GRAVITY
        accel_g  += self.np_random.normal(0, ACCEL_NOISE_STD, 3)
        accel_norm = np.clip(accel_g / 2.0, -1.0, 1.0).astype(np.float32)

        # Attitude error (roll & pitch, relative to setpoint)
        quat = self._sensor(self._quat_sensor_id, 4)
        roll, pitch, _ = _quat_to_euler(quat)
        roll_err  = np.float32(np.clip(math.degrees(roll  - self._cmd_roll)  / 180.0, -1.0, 1.0))
        pitch_err = np.float32(np.clip(math.degrees(pitch - self._cmd_pitch) / 180.0, -1.0, 1.0))

        # Altitude error (positive = above target, relative to setpoint)
        pos = self._sensor(self._pos_sensor_id, 3)
        alt_err = np.float32(np.clip((pos[2] - self._cmd_alt) / ALT_SCALE, -1.0, 1.0))

        # Vertical velocity
        vel = self._sensor(self._vel_sensor_id, 3)
        vvel = np.float32(np.clip(vel[2] / VVEL_SCALE, -1.0, 1.0))

        # Horizontal velocity (world frame X/Y) – new obs[10], obs[11]
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

        # --- All terms are dense, continuous, and observable ----------------

        # 1. Altitude hold: quadratic, ±2m range, dominates the reward
        alt_err = pos[2] - self._cmd_alt
        r_alt = -2.0 * alt_err * alt_err            # -2 at 1m off, -0.02 at 10cm

        # 2. Level orientation: quadratic tilt penalty (relative to setpoint)
        roll_dev  = roll  - self._cmd_roll
        pitch_dev = pitch - self._cmd_pitch
        tilt_sq = roll_dev * roll_dev + pitch_dev * pitch_dev   # radians²
        r_orient = -1.0 * tilt_sq                               # -0.62 at 45°

        # 3. Velocity penalty: stay still (XY + Z)
        r_vel = -0.1 * float(np.dot(vel, vel))      # quadratic, much smoother

        # 4. Angular velocity: penalise spin (quadratic)
        r_angvel = -0.01 * float(np.dot(ang_vel, ang_vel))

        # 5. Motor smoothness: penalise action variance
        r_smooth = -0.2 * float(np.var(action))

        # 6. Alive bonus: reward for not crashing
        r_alive = 2.0

        return r_alt + r_orient + r_vel + r_angvel + r_smooth + r_alive

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
