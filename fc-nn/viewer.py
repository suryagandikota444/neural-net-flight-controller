"""
viewer.py – Interactive MuJoCo viewer for the quadrotor.

Modes
-----
  python viewer.py                                # random actions (test the model file)
  python viewer.py --model runs/quad_hover/best_model.zip   # policy rollout
  python viewer.py --model runs/quad_hover/best_model.zip --record flight.mp4

Controller (gamepad / joystick – Mode 2 layout)
---------------
  Left  stick X  : Yaw rate    (±MAX_YAW_RATE_CMD deg/s)
  Left  stick Y  : Altitude    (hold centre = 1 m; pushed up/down ±1 m)
  Right stick X  : Roll        (±MAX_CMD_ANGLE deg)
  Right stick Y  : Pitch       (±MAX_CMD_ANGLE deg)

Keyboard fallback (when no gamepad detected – active as long as viewer is open):
  Up / Down    : Pitch forward / back
  Left / Right : Roll left / right
  Z / X        : Yaw left / right
  PgUp / PgDn  : Altitude up / down
  (all keys are momentary – release to return to centre; alt holds last value)

Controls (MuJoCo viewer window):
  Mouse drag   : orbit / pan
  Scroll       : zoom
  Space        : pause / resume
  Backspace    : reset episode
  Tab          : toggle overlay
  Esc / Q      : quit
"""

import argparse
import math
import time
import numpy as np

# ---------------------------------------------------------------------------
# Gamepad / joystick support (pygame).
# ---------------------------------------------------------------------------
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

# Deadzone applied to each raw axis value before use
_AXIS_DEADZONE = 0.08


def _apply_deadzone(v: float, dz: float = _AXIS_DEADZONE) -> float:
    """Return 0 if |v| < dz, else rescale to keep full range."""
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


class JoystickReader:
    """
    Thin wrapper around pygame joystick.

    Mode-2 layout (standard FPV):
      Axis 0  Left  X  → yaw rate
      Axis 1  Left  Y  → altitude offset  (push up = higher; pygame Y is inverted)
      Axis 2  Right X  → roll
      Axis 3  Right Y  → pitch            (push up = nose up; pygame Y is inverted)
    """

    # If your controller has a different mapping, change these indices.
    AXIS_YAW_RATE = 0
    AXIS_ALT      = 1
    AXIS_ROLL     = 2
    AXIS_PITCH    = 3

    def __init__(self):
        self._joy = None
        self._available = False
        if not _PYGAME_OK:
            print("[joystick] pygame not installed – run: pip install pygame")
            return
        pygame.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        if n == 0:
            print("[joystick] No joystick / gamepad detected – falling back to keyboard.")
            return
        self._joy = pygame.joystick.Joystick(0)
        self._joy.init()
        print(f"[joystick] Using: {self._joy.get_name()}  "
              f"({self._joy.get_numaxes()} axes, "
              f"{self._joy.get_numbuttons()} buttons)")
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def read(self,
             max_roll_deg: float  = 30.0,
             max_pitch_deg: float = 30.0,
             max_yaw_dps: float   = 90.0,
             base_alt: float      = 1.0,
             alt_range: float     = 1.0,
             ) -> dict:
        """
        Pump pygame events and return current setpoints.

        Returns
        -------
        dict with keys: roll_deg, pitch_deg, yaw_rate_dps, alt_m
        """
        if not self._available:
            return dict(roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0, alt_m=base_alt)

        pygame.event.pump()

        def _ax(idx):
            n = self._joy.get_numaxes()
            if idx >= n:
                return 0.0
            return _apply_deadzone(self._joy.get_axis(idx))

        yaw_raw   = _ax(self.AXIS_YAW_RATE)
        alt_raw   = _ax(self.AXIS_ALT)    # pygame Y up = -1  → invert
        roll_raw  = _ax(self.AXIS_ROLL)
        pitch_raw = _ax(self.AXIS_PITCH)  # pygame Y up = -1  → invert

        return dict(
            roll_deg      =  roll_raw  * max_roll_deg,
            pitch_deg     = -pitch_raw * max_pitch_deg,   # invert Y
            yaw_rate_dps  =  yaw_raw   * max_yaw_dps,
            alt_m         =  base_alt  - alt_raw * alt_range,  # invert Y
        )

    def close(self):
        if _PYGAME_OK:
            pygame.quit()


class KeyboardReader:
    """
    OS-level keyboard reader using pynput.
    Runs a background listener thread – works regardless of which window
    has focus, so pressing keys while the MuJoCo viewer is open is captured fine.

    Uses arrow keys + Z/X/PgUp/PgDn to avoid conflicting with MuJoCo's
    built-in viewer shortcuts (which consume W/A/S/D/Q/E/R/F etc.).

    Up/Down = pitch   Left/Right = roll   Z/X = yaw   PgUp/PgDn = altitude
    All keys are momentary except altitude (PgUp/PgDn accumulate).
    """

    def __init__(self,
                 max_roll_deg:  float = 20.0,
                 max_pitch_deg: float = 20.0,
                 max_yaw_dps:   float = 60.0,
                 base_alt:      float = 1.0,
                 alt_step:      float = 0.003):
        self.max_roll_deg  = max_roll_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_yaw_dps   = max_yaw_dps
        self._current_alt  = base_alt
        self.alt_step      = alt_step
        self._held         = set()   # string tags for currently held keys

        try:
            from pynput import keyboard as _kb
            self._Key = _kb.Key   # keep reference for special-key comparison
            self._listener = _kb.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()
            print("[keyboard] Up/Down=pitch  Left/Right=roll  Z/X=yaw  PgUp/PgDn=alt")
        except Exception as exc:
            print(f"[keyboard] pynput unavailable ({exc}) – keyboard control disabled.")
            self._listener = None
            self._Key = None

    def _tag(self, key) -> str | None:
        """Return a stable string tag for a key, or None to ignore it."""
        Key = self._Key
        if Key is None:
            return None
        # Special keys
        if   key == Key.up:        return 'up'
        elif key == Key.down:      return 'down'
        elif key == Key.left:      return 'left'
        elif key == Key.right:     return 'right'
        elif key == Key.page_up:   return 'pgup'
        elif key == Key.page_down: return 'pgdn'
        # Regular character keys
        try:
            c = key.char.lower()
            if c in ('z', 'x'):
                return c
        except AttributeError:
            pass
        return None

    def _on_press(self, key):
        tag = self._tag(key)
        if tag:
            self._held.add(tag)

    def _on_release(self, key):
        tag = self._tag(key)
        if tag:
            self._held.discard(tag)

    def read(self) -> dict:
        roll_deg  = 0.0
        pitch_deg = 0.0
        yaw_dps   = 0.0

        held = self._held
        if 'left'  in held:  roll_deg  -= self.max_roll_deg
        if 'right' in held:  roll_deg  += self.max_roll_deg
        if 'up'    in held:  pitch_deg += self.max_pitch_deg
        if 'down'  in held:  pitch_deg -= self.max_pitch_deg
        if 'z'     in held:  yaw_dps   -= self.max_yaw_dps
        if 'x'     in held:  yaw_dps   += self.max_yaw_dps
        if 'pgup'  in held:  self._current_alt = min(self._current_alt + self.alt_step, 3.0)
        if 'pgdn'  in held:  self._current_alt = max(self._current_alt - self.alt_step, 0.2)

        return dict(
            roll_deg     = roll_deg,
            pitch_deg    = pitch_deg,
            yaw_rate_dps = yaw_dps,
            alt_m        = self._current_alt,
        )

    def close(self):
        if self._listener is not None:
            self._listener.stop()


def _load_policy(model_path: str | None):
    """Return a policy function obs→action.  None = random."""
    if model_path is None:
        def random_policy(obs):
            return np.random.uniform(0, 1, size=4).astype(np.float32)
        return random_policy

    from stable_baselines3 import PPO
    import torch

    ppo = PPO.load(model_path)

    def sb3_policy(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = ppo.predict(obs_t, deterministic=True)
        return np.array(action, dtype=np.float32).flatten()

    return sb3_policy


def _overlay_text(env, obs, action, step, ep_reward, joy_sp: dict | None = None):
    """Build a compact HUD string."""
    pos  = env._sensor(env._pos_sensor_id, 3)
    quat = env._sensor(env._quat_sensor_id, 4)
    from drone_env import _quat_to_euler
    r, p, y = _quat_to_euler(quat)
    lines = [
        f"Step {step:5d}  |  Reward {ep_reward:+.2f}",
        f"Pos   x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:.3f}",
        f"RPY   r={math.degrees(r):+6.1f}°  p={math.degrees(p):+6.1f}°  y={math.degrees(y):+6.1f}°",
        f"Motor {action[0]:.2f}  {action[1]:.2f}  {action[2]:.2f}  {action[3]:.2f}",
    ]
    if joy_sp is not None:
        lines.append(
            f"CMD   r={joy_sp['roll_deg']:+5.1f}°  p={joy_sp['pitch_deg']:+5.1f}°  "
            f"yr={joy_sp['yaw_rate_dps']:+5.0f}°/s  alt={joy_sp['alt_m']:.2f}m"
        )
    return "\n".join(lines)


def run_viewer(model_path=None, record_path=None, max_episodes=0, slow_factor=1.0):
    from drone_env import QuadrotorEnv

    render_mode = "rgb_array" if record_path else "human"
    env = QuadrotorEnv(render_mode=render_mode)
    policy = _load_policy(model_path)

    # ---- Input (joystick preferred, keyboard fallback) ------------------
    joy = JoystickReader()
    if joy.available:
        active_reader = joy
    else:
        keyboard = KeyboardReader()
        active_reader = keyboard

    # Viewer controls setpoints manually – disable training randomisation
    env._randomize_setpoints = False
    env._max_ep_steps = 2000   # 20 s episodes in viewer

    # Exponential smoothing for setpoints so keyboard steps feel continuous
    SMOOTH_ALPHA = 0.08   # 0 = no change, 1 = instant; 0.08 ≈ ~80 ms rise
    smooth_sp = dict(roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0, alt_m=1.0)

    def _clean_reset():
        """Use the same consistent nominal-physics reset as evaluation."""
        smooth_sp.update(roll_deg=0.0, pitch_deg=0.0, yaw_rate_dps=0.0, alt_m=1.0)
        return env.reset(options={
            "domain_randomization": False,
            "randomize_setpoints": False,
            "initial_rpy_deg": [0.0, 0.0, 0.0],
            "setpoint": dict(smooth_sp),
        })

    frames = []
    episode  = 0
    ep_limit = max_episodes if max_episodes > 0 else float("inf")

    try:
        while episode < ep_limit:
            obs, _ = _clean_reset()
            ep_reward = 0.0
            step = 0
            terminated = truncated = False

            while not terminated and not truncated:
                # ---- Read input & smooth setpoint into env ------------
                raw_sp = active_reader.read()
                for k in smooth_sp:
                    smooth_sp[k] += SMOOTH_ALPHA * (raw_sp[k] - smooth_sp[k])
                joy_sp = smooth_sp   # alias for status prints
                env.set_setpoint(
                    roll_deg     = smooth_sp["roll_deg"],
                    pitch_deg    = smooth_sp["pitch_deg"],
                    yaw_rate_dps = smooth_sp["yaw_rate_dps"],
                    alt_m        = smooth_sp["alt_m"],
                )

                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                step += 1

                if render_mode == "human":
                    env.render()
                    if not env._viewer.is_running():
                        env.close()
                        active_reader.close()
                        return
                    # Each policy step = 10 ms sim time; sleep to match real-time
                    time.sleep(0.01 * slow_factor)
                    # Periodic status so you can verify keyboard inputs are working
                    if step % 50 == 0:
                        pos = env._sensor(env._pos_sensor_id, 3)
                        print(f"\r  cmd: r={joy_sp['roll_deg']:+5.1f}°  "
                              f"p={joy_sp['pitch_deg']:+5.1f}°  "
                              f"alt={joy_sp['alt_m']:.2f}m  |  "
                              f"z={pos[2]:.2f}m  "
                              f"mot=[{action[0]:.2f} {action[1]:.2f} {action[2]:.2f} {action[3]:.2f}]",
                              end="", flush=True)
                    # Print full obs vector + NN output every ~1 s (100 steps × 10 ms)
                    if step % 100 == 0:
                        obs_str = ",".join(f"{v:+.4f}" for v in obs)
                        act_str = ",".join(f"{v:+.4f}" for v in action)
                        print(f"\n  [t={step*0.01:.1f}s] obs=[{obs_str}]")
                        print(f"          act=[{act_str}]")
                else:
                    # Record every ~30th frame (for sane video file sizes)
                    if step % max(1, int(1.0 / (30 * 0.00025))) == 0:
                        frames.append(env.render())

            episode += 1
            print(f"[viewer] Episode {episode:3d}  steps={step:5d}  "
                  f"reward={ep_reward:+8.2f}  "
                  f"{'CRASH' if terminated else 'TIMEOUT'}")

    except KeyboardInterrupt:
        print("\n[viewer] Interrupted.")

    env.close()
    active_reader.close()

    if record_path and frames:
        _save_video(frames, record_path)


def _save_video(frames, path, fps=30):
    """Save a list of RGB arrays as an MP4 video."""
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p")
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[viewer] Saved video to {path}  ({len(frames)} frames)")
    except ImportError:
        print("[viewer] Install imageio + imageio-ffmpeg to save videos:")
        print("         pip install imageio imageio-ffmpeg")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo quadrotor viewer."
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a trained SB3 PPO model .zip. "
                             "Omit for random actions.")
    parser.add_argument("--record", type=str, default=None,
                        help="Path to save an MP4 recording (e.g. flight.mp4).")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run (0 = infinite).")
    parser.add_argument("--slow", type=float, default=1.0,
                        help="Slow-motion factor. 2.0 = half speed.")
    args = parser.parse_args()

    print("[viewer] Launching MuJoCo viewer ...")
    if args.model:
        print(f"         Policy: {args.model}")
    else:
        print("         Policy: RANDOM (no model loaded)")
    if args.record:
        print(f"         Recording to: {args.record}")

    run_viewer(
        model_path=args.model,
        record_path=args.record,
        max_episodes=args.episodes,
        slow_factor=args.slow,
    )


if __name__ == "__main__":
    main()
