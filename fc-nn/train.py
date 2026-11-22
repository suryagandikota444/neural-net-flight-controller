"""
train.py  –  PPO training entry-point for the quadrotor hover task.

Usage:
    python train.py --save_dir runs/contract_s1 --seed 1            # firmware contract (default)
    python train.py --save_dir runs/smoke --timesteps 50000         # quick smoke-test
    python train.py --save_dir runs/legacy_x --legacy               # pre-2026-11-21 env
    python train.py --save_dir runs/contract_s1b --resume runs/contract_s1/best_model.zip

The env runs with firmware_contract=True unless --legacy is given; override
single contract entries with --contract '{"ground_start_frac": 0.0}'.
Every run writes <save_dir>/run_config.json (args, seed, full contract config,
PPO settings, git commit, file hashes, package versions) and
<save_dir>/source.patch (git diff HEAD) so it can be reproduced.
The save_dir must be new (or pass --overwrite).

Visualisation:
    TensorBoard logs go to <save_dir>/tensorboard.  Launch with:
        tensorboard --logdir runs
    then open http://localhost:6006
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

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

from drone_env import QuadrotorEnv, MAX_EP_STEPS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

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
    tensorboard_log="runs/tensorboard",   # overridden per run: <save_dir>/tensorboard
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
        self._reasons = {}

    def _on_step(self) -> bool:
        # Collect episode stats from the info buffer
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
                # If ep length < MAX_EP_STEPS, it was a crash/termination
                self._ep_crashes.append(1.0 if ep["l"] < MAX_EP_STEPS else 0.0)
                reason = info.get("termination_reason")
                if reason:
                    self._reasons[reason] = self._reasons.get(reason, 0) + 1

        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return
        self.logger.record("drone/mean_reward",  np.mean(self._ep_rewards))
        self.logger.record("drone/mean_ep_len",   np.mean(self._ep_lengths))
        self.logger.record("drone/crash_rate",     np.mean(self._ep_crashes))
        n = len(self._ep_lengths)
        for reason in ("ground_contact", "tilt_limit", "altitude_limit", "no_liftoff"):
            self.logger.record(f"drone/term_{reason}", self._reasons.get(reason, 0) / n)
        self._reasons.clear()
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
                 verbose=0, start=0.0, state_path=None):
        super().__init__(verbose)
        self._threshold = advance_threshold
        self._step_size = step_size
        self._window = window
        self._crash_history = []
        self._progress = float(start)
        self._state_path = state_path

    def _on_training_start(self) -> None:
        # Resumed runs continue from the saved progress instead of 0.
        self.training_env.env_method("set_curriculum", self._progress)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._crash_history.append(1.0 if ep["l"] < MAX_EP_STEPS else 0.0)
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
        if self._state_path:
            Path(self._state_path).write_text(json.dumps(
                {"progress": self._progress, "num_timesteps": int(self.num_timesteps)}))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total environment steps to train for.")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="Parallel environments (defaults to CPU count).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds PPO, torch and every env (env i gets seed+i).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model zip to resume from.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="NEW directory for checkpoints, best model, logs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow a save_dir that already holds a run.")
    parser.add_argument("--legacy", action="store_true",
                        help="Train in the pre-2026-11-21 env (firmware_contract=False).")
    parser.add_argument("--contract", type=str, default=None,
                        help="JSON dict overriding FIRMWARE_CONTRACT_DEFAULTS entries.")
    parser.add_argument("--curriculum_start", type=float, default=None,
                        help="Initial curriculum progress (default 0, or the value "
                             "saved next to --resume's model).")
    parser.add_argument("--checkpoint_freq", type=int, default=500_000,
                        help="Save a checkpoint every N total timesteps.")
    parser.add_argument("--eval_freq", type=int, default=100_000,
                        help="EvalCallback every N total timesteps.")
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--final_eval", choices=["full", "quick", "none"], default="full",
                        help="Run eval_contract.py on the best model at the end.")
    parser.add_argument("--no_progress_bar", action="store_true")
    return parser.parse_args()


def _git(*cmd, raw=False):
    try:
        out = subprocess.check_output(["git", *cmd], cwd=REPO, stderr=subprocess.DEVNULL)
        return out if raw else out.decode("utf-8", errors="replace")
    except Exception as exc:          # not a git checkout / git missing
        msg = f"<git {' '.join(cmd)} failed: {exc}>"
        return msg.encode() if raw else msg


def write_run_manifest(args, save_dir, env_config, n_envs, curriculum_start):
    """Everything needed to reproduce this run, written before training starts."""
    tracked = sorted({*HERE.glob("*.py"), HERE / "quadrotor.xml",
                      REPO / "fc_config.h", REPO / "main.c", REPO / "baro.c",
                      REPO / "nn.h", REPO / "weights.h"})
    import gymnasium, mujoco, stable_baselines3
    manifest = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        argv=sys.argv, args=vars(args), seed=args.seed, n_envs=n_envs,
        curriculum_start=curriculum_start,
        env_config=env_config,
        ppo_kwargs={k: v for k, v in PPO_KWARGS.items()},
        policy_kwargs=dict(net_arch=POLICY_KWARGS["net_arch"],
                           activation_fn=POLICY_KWARGS["activation_fn"].__name__),
        resume_from=(dict(path=str(Path(args.resume).resolve()),
                          sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest())
                     if args.resume else None),
        git=dict(commit=_git("rev-parse", "HEAD").strip(),
                 status=_git("status", "--porcelain").splitlines()),
        file_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in tracked if p.exists()},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      torch=torch.__version__, mujoco=mujoco.__version__,
                      gymnasium=gymnasium.__version__,
                      stable_baselines3=stable_baselines3.__version__,
                      platform=platform.platform()),
    )
    (save_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, default=str))
    # Tracked changes vs the commit (binary-safe) ...
    (save_dir / "source.patch").write_bytes(_git("diff", "HEAD", raw=True))
    # ... plus a copy of every source that defines the task, because untracked
    # files (e.g. fc_config.h, new scripts) are not in `git diff`.
    snap = save_dir / "source_snapshot"
    for p in tracked:
        if p.exists():
            dst = snap / p.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
    return manifest


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not args.overwrite:
        sys.exit(f"[train] {save_dir} already exists and is not empty — pick a new "
                 f"--save_dir (or pass --overwrite).")
    save_dir.mkdir(parents=True, exist_ok=True)

    firmware_contract = False if args.legacy else (json.loads(args.contract) if args.contract else True)
    env_kwargs = {"render_mode": None, "firmware_contract": firmware_contract}
    probe = QuadrotorEnv(**env_kwargs)          # validates the config, records it
    env_config = probe.contract_config()
    probe.close()

    curriculum_start = args.curriculum_start
    if curriculum_start is None and args.resume:
        state = Path(args.resume).resolve().parent / "curriculum.json"
        curriculum_start = json.loads(state.read_text())["progress"] if state.exists() else 0.0
    curriculum_start = float(curriculum_start or 0.0)

    n_envs = args.n_envs or max(1, os.cpu_count() or 1)
    print(f"[train] Using {n_envs} parallel environments  (MuJoCo backend)")
    print(f"[train] Total timesteps: {args.timesteps:,}   seed: {args.seed}")
    print(f"[train] Env: {'LEGACY' if args.legacy else 'firmware contract'}  "
          f"curriculum start {curriculum_start:.2f}")
    print(f"[train] TensorBoard:  tensorboard --logdir {save_dir.parent}")
    write_run_manifest(args, save_dir, env_config, n_envs, curriculum_start)

    # --- Vectorised training environment (headless) -----------------------
    vec_env = make_vec_env(
        QuadrotorEnv,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
    )

    # --- Separate eval environment (single, no subprocess) ----------------
    # The eval env is never sent curriculum updates, so it must not phase the
    # actuator delay in (that would pick best_model by no-delay performance):
    # it always uses the full measured delay range.
    eval_kwargs = dict(env_kwargs)
    if eval_kwargs.get("firmware_contract") is not False:
        fc = eval_kwargs.get("firmware_contract")
        eval_kwargs["firmware_contract"] = {**(fc if isinstance(fc, dict) else {}),
                                            "actuator_curriculum": False}
    eval_env = make_vec_env(QuadrotorEnv, n_envs=1, seed=args.seed + 10_000,
                            env_kwargs=eval_kwargs)

    # --- Callbacks --------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="quad",
        save_replay_buffer=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    drone_cb = DroneMetricsCallback()
    curric_cb = CurriculumCallback(
        advance_threshold=0.30,   # advance when <30% of episodes crash
        step_size=0.02,           # 50 advances to reach 1.0 (gradual ramp)
        window=200,               # rolling window of 200 episodes
        verbose=1,
        start=curriculum_start,
        state_path=save_dir / "curriculum.json",
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb, drone_cb, curric_cb])

    # --- Model ------------------------------------------------------------
    ppo_kwargs = {**PPO_KWARGS, "tensorboard_log": str(save_dir / "tensorboard"),
                  "seed": args.seed, "device": "cpu"}
    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, **ppo_kwargs)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=POLICY_KWARGS,
            **ppo_kwargs,
        )

    # --- Train ------------------------------------------------------------
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=not args.no_progress_bar,
        tb_log_name=save_dir.name,
        reset_num_timesteps=not args.resume,
    )

    # Save final model
    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"[train] Final model saved to {final_path}.zip")

    vec_env.close()
    eval_env.close()

    # --- Firmware-contract acceptance gates on the selected model ----------
    if args.final_eval != "none" and not args.legacy:
        best = save_dir / "best_model.zip"
        target = best if best.exists() else Path(f"{final_path}.zip")
        cmd = [sys.executable, str(HERE / "eval_contract.py"), str(target),
               "--out", str(save_dir / f"contract_eval_{target.stem}.json")]
        if args.final_eval == "quick":
            cmd.append("--quick")
        print("[train] " + " ".join(cmd))
        subprocess.run(cmd, cwd=HERE, check=False)


if __name__ == "__main__":
    main()
