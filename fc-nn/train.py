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
    n_steps=2048,               # steps per env before update
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,              # small entropy bonus to keep exploring
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
                self._ep_crashes.append(1.0 if ep["l"] < 500 else 0.0)

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
    callbacks = CallbackList([checkpoint_cb, eval_cb, drone_cb])

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
