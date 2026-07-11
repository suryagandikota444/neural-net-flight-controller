"""
train.py  –  PPO training entry-point for the quadrotor hover task.

Usage:
    python train.py                        # full 10 M step run
    python train.py --timesteps 500000     # quick smoke-test
    python train.py --resume runs/quad_hover/best_model

Visualisation:
    TensorBoard logs are written to runs/tensorboard/ by default.
    Launch with:
        tensorboard --logdir runs/tensorboard
    then open http://localhost:6006
"""

import argparse
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from drone_env import QuadrotorEnv

# ---------------------------------------------------------------------------
# Network architecture must match the STM32 weights.h layout exactly:
#   input(12) -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(4)
# ---------------------------------------------------------------------------
POLICY_KWARGS = dict(
    net_arch=[32, 32],          # two hidden layers of 32 neurons
    activation_fn=torch.nn.ReLU,
)

# ---------------------------------------------------------------------------
# PPO hyper-parameters (tuned for hover task)
# ---------------------------------------------------------------------------
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=4096,               # longer rollouts → better value estimates
    batch_size=512,
    n_epochs=10,
    gamma=0.995,                # higher discount → patient, stable behaviour
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,             # less exploration noise → more precise control
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="runs/tensorboard",
)


# ---------------------------------------------------------------------------
# Custom TensorBoard callback  – logs extra drone-specific scalars
# ---------------------------------------------------------------------------

class DroneMetricsCallback(BaseCallback):
    """
    Logs per-rollout drone metrics to TensorBoard:
        - mean episode reward
        - mean episode length
        - crash rate (fraction of episodes that terminated early)
        - mean altitude at episode end
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_rewards = []
        self._ep_lengths = []
        self._ep_crashes = []

    def _on_step(self) -> bool:
        # Collect episode stats from the info buffer
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
                # If ep length < MAX_EP_STEPS, it was a crash/termination
                self._ep_crashes.append(1.0 if ep["l"] < 1000 else 0.0)

        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return
        self.logger.record("drone/mean_reward",  np.mean(self._ep_rewards))
        self.logger.record("drone/mean_ep_len",   np.mean(self._ep_lengths))
        self.logger.record("drone/crash_rate",     np.mean(self._ep_crashes))
        self._ep_rewards.clear()
        self._ep_lengths.clear()
        self._ep_crashes.clear()


# ---------------------------------------------------------------------------
# Curriculum callback — advances difficulty based on rolling crash rate
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Tracks rolling crash rate and advances curriculum progress when the agent
    is performing well (low crash rate).  Broadcasts to all SubprocVecEnv workers.

    Schedule:
        - progress stays at 0 until the agent reaches `advance_threshold` crash rate
        - then advances by `step_size` each time the rolling crash rate is checked
        - never goes backwards (monotonic ramp)
        - logs curriculum/progress and curriculum/tilt_max to TensorBoard

    Parameters
    ----------
    advance_threshold : float
        Crash rate must be BELOW this to advance (default 0.30 = 30% crashes).
    step_size : float
        How much to increment progress each rollout when threshold is met.
    window : int
        Number of episodes to average over for the rolling crash rate.
    """

    def __init__(self, advance_threshold=0.30, step_size=0.02, window=200,
                 verbose=0):
        super().__init__(verbose)
        self._threshold = advance_threshold
        self._step_size = step_size
        self._window = window
        self._crash_history = []
        self._progress = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._crash_history.append(1.0 if ep["l"] < 1000 else 0.0)
                # Keep a bounded window
                if len(self._crash_history) > self._window * 2:
                    self._crash_history = self._crash_history[-self._window:]
        return True

    def _on_rollout_end(self) -> None:
        if len(self._crash_history) < 50:
            # Not enough data yet — keep at current progress
            return

        rolling_crash = np.mean(self._crash_history[-self._window:])

        # Advance if crash rate is below threshold (agent has mastered current level)
        if rolling_crash < self._threshold and self._progress < 1.0:
            self._progress = min(1.0, self._progress + self._step_size)

            # Broadcast new curriculum to all envs (works with SubprocVecEnv)
            self.training_env.env_method("set_curriculum", self._progress)

            if self.verbose:
                from drone_env import CURRIC_TILT_START, CURRIC_TILT_END
                _tilt = CURRIC_TILT_START + (CURRIC_TILT_END - CURRIC_TILT_START) * self._progress
                print(f"[curriculum] progress={self._progress:.2f}  "
                      f"crash_rate={rolling_crash:.3f}  tilt_max={_tilt:.1f}°")

        # Log to TensorBoard
        from drone_env import CURRIC_TILT_START, CURRIC_TILT_END, CURRIC_CMD_ANGLE_START, CURRIC_CMD_ANGLE_END
        self.logger.record("curriculum/progress", self._progress)
        self.logger.record("curriculum/tilt_max_deg",
                           CURRIC_TILT_START + (CURRIC_TILT_END - CURRIC_TILT_START) * self._progress)
        self.logger.record("curriculum/cmd_angle_deg",
                           CURRIC_CMD_ANGLE_START + (CURRIC_CMD_ANGLE_END - CURRIC_CMD_ANGLE_START) * self._progress)
        self.logger.record("curriculum/rolling_crash_rate",
                           np.mean(self._crash_history[-self._window:]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total environment steps to train for.")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="Parallel environments (defaults to CPU count).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model zip to resume from.")
    parser.add_argument("--save_dir", type=str, default="runs/quad_hover",
                        help="Directory to write checkpoints and best model.")
    return parser.parse_args()


def main():
    args = parse_args()

    n_envs = args.n_envs or max(1, os.cpu_count() or 1)
    print(f"[train] Using {n_envs} parallel environments  (MuJoCo backend)")
    print(f"[train] Total timesteps: {args.timesteps:,}")
    print(f"[train] TensorBoard:  tensorboard --logdir runs/tensorboard")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Vectorised training environment (headless) -----------------------
    vec_env = make_vec_env(
        QuadrotorEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": None},
    )

    # --- Separate eval environment (single, no subprocess) ----------------
    eval_env = make_vec_env(QuadrotorEnv, n_envs=1,
                            env_kwargs={"render_mode": None})

    # --- Callbacks --------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="quad",
        save_replay_buffer=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=os.path.join(args.save_dir, "eval_logs"),
        eval_freq=max(100_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    drone_cb = DroneMetricsCallback()
    curric_cb = CurriculumCallback(
        advance_threshold=0.30,   # advance when <30% of episodes crash
        step_size=0.02,           # 50 advances to reach 1.0 (gradual ramp)
        window=200,               # rolling window of 200 episodes
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb, drone_cb, curric_cb])

    # --- Model ------------------------------------------------------------
    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, **PPO_KWARGS)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=POLICY_KWARGS,
            **PPO_KWARGS,
        )

    # --- Train ------------------------------------------------------------
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"[train] Final model saved to {final_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()