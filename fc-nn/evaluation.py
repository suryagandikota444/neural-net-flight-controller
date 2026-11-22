"""Seeded, failure-aware attitude/altitude benchmarks (no Torch dependency).

SCENARIOS          — legacy suite (target 1.0 m, true hvel); evaluate old
                     checkpoints with env_factory=lambda: QuadrotorEnv(firmware_contract=False).
CONTRACT_SCENARIOS — firmware-contract suite (target 0.3 m above the baro
                     ground reference, obs[10:12] = 0, baro + P+D loop).
The altitude target is read from the env (env.target_z), so one scorer
serves both.
"""
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
    # --- firmware-contract additions (defaults keep the legacy behaviour) ---
    ground_start: bool = False          # arm on the floor, main.c throttle ramp
    initial_alt_agl: float = None       # start height above ground ref (contract env)
    thrust_scale: float = None          # collective thrust-map error (contract env)
    score_after_s: float = 0.0          # ignore the take-off transient
    max_mean_lean_deg: float = None     # |time-mean roll| and |mean pitch| limit
    max_mean_yaw_rate_deg_s: float = None  # |time-mean yaw rate| limit (steady spin)
    advisory: bool = False              # reported, not part of the pass verdict

    def reset_options(self):
        options = dict(domain_randomization=self.randomized,
                       randomize_setpoints=False, setpoint={"alt_m": 1.0},
                       max_steps=round(self.seconds / POLICY_DT))
        if not self.ground_start:
            options["initial_rpy_deg"] = self.tilt
        if not self.randomized:
            options["wind_force"] = self.wind
        # Contract-only options (the legacy env ignores unknown keys)
        options["ground_start"] = self.ground_start
        if self.initial_alt_agl is not None:
            options["initial_alt_agl"] = self.initial_alt_agl
        if self.thrust_scale is not None:
            options["thrust_scale"] = self.thrust_scale
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

# Firmware-contract acceptance suite.  No position/velocity sensing, so
# horizontal drift is reported but not gated (envelope 1 km); a steady lean is
# what causes drift, and that IS gated.  Tilt recoveries start 0.5 m above the
# ground reference because ground contact ends the episode.
_C = dict(max_mean_yaw_rate_deg_s=5.0,
          altitude_band_m=0.10, altitude_rmse_limit_m=0.10, time_in_band_min=0.8,
          attitude_rms_limit_deg=3.0, max_mean_lean_deg=1.0, max_drift_m=1000.0)
_TILT = dict(initial_alt_agl=0.5, max_altitude_loss_m=0.4, max_drift_m=1000.0,
             recovery_deadline_s=2.0)
CONTRACT_SCENARIOS = (
    Scenario("hover30", 30, **_C),
    Scenario("takeoff20", 20, ground_start=True, score_after_s=5.0, **_C),
    Scenario("roll_plus20", 5, tilt=(20, 0, 0), **_TILT),
    Scenario("roll_minus20", 5, tilt=(-20, 0, 0), **_TILT),
    Scenario("pitch_plus20", 5, tilt=(0, 20, 0), **_TILT),
    Scenario("pitch_minus20", 5, tilt=(0, -20, 0), **_TILT),
    Scenario("randomized_hover30", 30, randomized=True, initial_alt_agl=0.3,
             **{**_C, "attitude_rms_limit_deg": 4.0, "max_mean_lean_deg": 2.0,
                "altitude_rmse_limit_m": 0.15, "altitude_band_m": 0.15}),
    Scenario("wind10", 10, wind=(0.5, 0, 0),
             **{**_C, "attitude_rms_limit_deg": 5.0, "max_mean_lean_deg": None}),
    Scenario("thrust_weak25", 20, thrust_scale=0.75, advisory=True,
             **{**_C, "altitude_rmse_limit_m": 0.15, "altitude_band_m": 0.15}),
    Scenario("thrust_strong25", 20, thrust_scale=1.25, advisory=True,
             **{**_C, "altitude_rmse_limit_m": 0.15, "altitude_band_m": 0.15}),
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
                     motor_commands, *, terminated=False, reason=None, target_z=1.0):
    """Pure metric function; failures are reported separately from short-run RMSE.

    target_z: world z of the altitude target (legacy env 1.0; contract env
    GROUND_REST_Z + 0.3).  Samples before scenario.score_after_s are excluded
    from the hover metrics (not from completion/termination)."""
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
    if scenario.score_after_s > 0:
        keep = times >= scenario.score_after_s
        if not np.any(keep):
            result.update(passed=False, failures=failures + ["no_scored_samples"])
            return result
        times, positions, attitudes_deg = times[keep], positions[keep], attitudes_deg[keep]
        angular_rates_deg, motor_commands = angular_rates_deg[keep], motor_commands[keep]
        intervals = intervals[keep]
    error = positions[:, 2] - target_z
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
    mean_roll = weighted(attitudes_deg[:, 0])
    mean_pitch = weighted(attitudes_deg[:, 1])
    mean_yaw_rate = weighted(angular_rates_deg[:, 2])
    if drift > scenario.max_drift_m: failures.append("horizontal_envelope")
    if (scenario.max_mean_lean_deg is not None and
            max(abs(mean_roll), abs(mean_pitch)) > scenario.max_mean_lean_deg):
        failures.append("steady_lean")
    if (scenario.max_mean_yaw_rate_deg_s is not None and
            abs(mean_yaw_rate) > scenario.max_mean_yaw_rate_deg_s):
        failures.append("yaw_spin")
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
                  attitude_rms_deg=att_rms, mean_roll_deg=mean_roll,
                  mean_pitch_deg=mean_pitch, mean_yaw_rate_deg_s=mean_yaw_rate,
                  max_drift_m=drift,
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
        target_z = float(getattr(env, "target_z", 1.0))
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
                   env.last_motor_cmd.copy(), previous_obs, action)
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
                                  data["motor_commands"], terminated=terminated, reason=reason,
                                  target_z=target_z)
        result.update(seed=seed, scenario=scenario.name, advisory=scenario.advisory,
                      target_z=target_z,
                      firmware_contract=bool(getattr(env, "firmware_contract", False)))
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
        summary[name]["advisory"] = bool(rows[0].get("advisory", False))
        for metric in ("altitude_rmse_m", "attitude_rms_deg", "mean_roll_deg",
                       "mean_pitch_deg", "mean_yaw_rate_deg_s", "max_drift_m"):
            vals = [r[metric] for r in complete if r.get(metric) is not None]
            summary[name]["completed_mean_"+metric] = float(np.mean(vals)) if vals else None
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
