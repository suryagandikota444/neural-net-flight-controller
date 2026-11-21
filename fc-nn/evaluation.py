"""Seeded, failure-aware attitude/altitude benchmarks (no Torch dependency)."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
from drone_env import QuadrotorEnv, DT, FRAMESKIP, _quat_to_euler

POLICY_DT = DT * FRAMESKIP


@dataclass(frozen=True)
class Scenario:
    name: str
    seconds: float
    tilt: tuple = (0.0, 0.0, 0.0)
    wind: tuple = (0.0, 0.0, 0.0)
    randomized: bool = False
    altitude_band_m: float = 0.05
    altitude_rmse_limit_m: float = 0.05
    time_in_band_min: float = 0.9
    attitude_rms_limit_deg: float = 2.0
    max_drift_m: float = 2.0  # test-area envelope; this is not position control
    recovery_deadline_s: float = 2.0
    recovery_dwell_s: float = 1.0
    max_altitude_loss_m: float = 0.5

    def reset_options(self):
        options = dict(domain_randomization=self.randomized,
                       randomize_setpoints=False, setpoint={"alt_m": 1.0},
                       initial_rpy_deg=self.tilt,
                       max_steps=round(self.seconds / POLICY_DT))
        if not self.randomized:
            options["wind_force"] = self.wind
        return options


SCENARIOS = (
    Scenario("hover30", 30),
    Scenario("roll_plus30", 5, tilt=(30, 0, 0)),
    Scenario("roll_minus30", 5, tilt=(-30, 0, 0)),
    Scenario("pitch_plus30", 5, tilt=(0, 30, 0)),
    Scenario("pitch_minus30", 5, tilt=(0, -30, 0)),
    Scenario("wind10", 10, wind=(0.5, 0, 0), altitude_band_m=0.1,
             altitude_rmse_limit_m=0.1, time_in_band_min=0.8, attitude_rms_limit_deg=5),
    Scenario("randomized_hover30", 30, randomized=True),
)


def first_sustained_time(times, in_band, dwell_s, dt=POLICY_DT):
    """Start of first completed dwell interval; never credit a single crossing."""
    count = 0
    required = max(1, math.ceil(dwell_s / dt - 1e-9))
    for i, good in enumerate(in_band):
        count = count + 1 if good else 0
        if count >= required:
            return float(times[i - required + 1])
    return None


def score_trajectory(scenario, times, positions, attitudes_deg, angular_rates_deg,
                     motor_commands, *, terminated=False, reason=None):
    """Pure metric function; failures are reported separately from short-run RMSE."""
    times = np.asarray(times)
    positions = np.asarray(positions)
    attitudes_deg = np.asarray(attitudes_deg)
    angular_rates_deg = np.asarray(angular_rates_deg)
    motor_commands = np.asarray(motor_commands)
    n = len(times)
    finite = n > 0 and all(np.all(np.isfinite(x)) for x in
                          (times, positions, attitudes_deg, angular_rates_deg, motor_commands))
    full_duration = finite and times[-1] >= scenario.seconds - POLICY_DT * 0.1
    failures = []
    if terminated: failures.append(reason or "terminated")
    if not finite: failures.append("nonfinite_or_empty")
    if not full_duration: failures.append("incomplete_duration")
    result = dict(steps=n, seconds=float(times[-1]) if n else 0.0,
                  completed=bool(full_duration and not terminated), termination_reason=reason)
    if not finite:
        result.update(passed=False, failures=failures)
        return result
    intervals = np.diff(np.r_[0.0, times])
    error = positions[:, 2] - 1.0
    weighted = lambda a: float(np.average(a, weights=intervals))
    alt_rmse = math.sqrt(weighted(error**2))
    att_rms = math.sqrt(weighted(np.sum(attitudes_deg[:, :2]**2, axis=1)))
    drift = float(np.max(np.linalg.norm(positions[:, :2], axis=1)))
    in_band = weighted(np.abs(error) < scenario.altitude_band_m)
    recovering = bool(any(scenario.tilt))
    recovery_mask = ((np.max(np.abs(attitudes_deg[:, :2]), axis=1) < 10) &
                     (np.max(np.abs(angular_rates_deg[:, :2]), axis=1) < 20) &
                     (np.abs(error) < 0.25))
    recovery = first_sustained_time(times, recovery_mask, scenario.recovery_dwell_s)
    altitude_loss = max(0.0, float(-np.min(error)))
    saturation = np.any((motor_commands <= 1e-6) | (motor_commands >= 1-1e-6), axis=1)
    longest = current = 0.0
    for saturated, dt in zip(saturation, intervals):
        current = current + dt if saturated else 0.0
        longest = max(current, longest)
    if drift > scenario.max_drift_m: failures.append("horizontal_envelope")
    if recovering:
        if recovery is None or recovery > scenario.recovery_deadline_s:
            failures.append("sustained_recovery")
        if altitude_loss > scenario.max_altitude_loss_m: failures.append("altitude_loss")
    else:
        if alt_rmse > scenario.altitude_rmse_limit_m: failures.append("altitude_rmse")
        if in_band < scenario.time_in_band_min: failures.append("time_in_altitude_band")
        if att_rms > scenario.attitude_rms_limit_deg: failures.append("attitude_rms")
    result.update(altitude_rmse_m=alt_rmse, altitude_bias_m=weighted(error),
                  max_altitude_error_m=float(np.max(np.abs(error))),
                  altitude_loss_m=altitude_loss, time_in_altitude_band=in_band,
                  attitude_rms_deg=att_rms, max_drift_m=drift,
                  recovery_s=recovery, saturation_fraction=weighted(saturation),
                  longest_saturation_s=longest, passed=not failures, failures=failures)
    return result


def run_episode(policy, scenario, seed, *, env_factory=QuadrotorEnv, trajectory_path=None):
    env = env_factory()
    data = {k: [] for k in ("time_s", "position_m", "attitude_deg", "angular_rate_deg_s",
                            "motor_commands", "observation", "raw_action")}
    terminated = False
    reason = None
    try:
        obs, reset_info = env.reset(seed=seed, options=scenario.reset_options())
        start = float(env._data.time)
        for _ in range(round(scenario.seconds / POLICY_DT)):
            action = np.asarray(policy(obs), dtype=np.float32).reshape(-1)
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                terminated, reason = True, "invalid_action"
                break
            previous_obs = obs.copy()
            obs, _, terminated, truncated, info = env.step(action)
            q = env._data.qpos.copy()
            angles = np.degrees(_quat_to_euler(q[3:7]))
            row = (env._data.time-start, q[:3], angles, np.degrees(env._data.qvel[3:6]),
                   env._data.ctrl[env._act_ids].copy(), previous_obs, action)
            for key, value in zip(data, row): data[key].append(value)
            if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(q)):
                terminated, reason = True, "nonfinite_state"
            if terminated:
                reason = reason or info.get("termination_reason") or "terminated"
                break
            if truncated: break
            env.render()
        result = score_trajectory(scenario, data["time_s"], data["position_m"],
                                  data["attitude_deg"], data["angular_rate_deg_s"],
                                  data["motor_commands"], terminated=terminated, reason=reason)
        result.update(seed=seed, scenario=scenario.name)
        result["plant"] = {"mass_kg": float(reset_info.get("mass_kg", 0)),
                           "cog_offset_m": np.asarray(reset_info.get("cog_offset_m", [0,0,0])).tolist()}
        if trajectory_path:
            path = Path(trajectory_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: np.asarray(v) for k,v in data.items()})
        return result
    finally:
        env.close()


def summarize(results):
    """Keep completion/pass counts alongside conditional completed-run metrics."""
    summary = {}
    for name in dict.fromkeys(r["scenario"] for r in results):
        rows = [r for r in results if r["scenario"] == name]
        complete = [r for r in rows if r["completed"]]
        summary[name] = dict(episodes=len(rows), completed=len(complete),
                             passed=sum(r["passed"] for r in rows))
        for metric in ("altitude_rmse_m", "attitude_rms_deg", "max_drift_m"):
            summary[name]["completed_mean_"+metric] = (
                float(np.mean([r[metric] for r in complete])) if complete else None)
        summary[name]["failure_counts"] = {reason: sum(reason in r["failures"] for r in rows)
            for reason in sorted({f for r in rows for f in r["failures"]})}
    return summary


def evaluate(policy, seeds=range(100,105), scenarios=SCENARIOS, *, env_factory=QuadrotorEnv,
             output=None, trajectories=None):
    results = []
    for scenario in scenarios:
        for seed in seeds:
            path = Path(trajectories)/f"{scenario.name}_{seed}.npz" if trajectories else None
            results.append(run_episode(policy, scenario, seed, env_factory=env_factory, trajectory_path=path))
        print(scenario.name, json.dumps(summarize(results)[scenario.name]), flush=True)
        if output:
            payload = dict(criteria=[asdict(s) for s in scenarios], episodes=results, summary=summarize(results))
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(payload, indent=2, allow_nan=False))
    return results
