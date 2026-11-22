"""Firmware-contract tests for drone_env.QuadrotorEnv(firmware_contract=True).

  * obs[10:12] == 0 on every step; obs[8] within ±ALT_OBS_CLAMP on every step
  * the main.c P+D altitude term really changes the applied motor command
    and the resulting vertical acceleration; the H2 throttle ramp matches main.c
  * barometer model == fw_common.BaroModel == baro.c semantics
  * constants agree with ../fc_config.h, ../baro.c, ../main.c, fw_common.FW
  * firmware_contract=False reproduces the pre-contract env bit-for-bit
    (fixtures/legacy_obs_golden.npz, recorded from the old drone_env.py)

Run:  python test_contract.py
"""
import math
import re
import unittest
from pathlib import Path

import mujoco
import numpy as np

import drone_env as de
from drone_env import (QuadrotorEnv, HOVER_THROTTLE, ACTION_SCALE, GROUND_REST_Z,
                       FIRMWARE_CONTRACT_DEFAULTS, MAX_THRUST)
from fw_common import FW, BaroModel, FirmwareController
from fixtures.make_legacy_golden import record

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NOISELESS = dict(domain_randomization=False, sensor_noise=False)


def c_define(path, name):
    m = re.search(rf"#define\s+{name}\s+\(?\s*([-+0-9.eE]+)f?", (REPO / path).read_text())
    assert m, f"{name} not found in {path}"
    return float(m.group(1))


class ObservationContract(unittest.TestCase):
    def test_hvel_zero_and_alt_clamped_every_step(self):
        env = QuadrotorEnv()
        rng = np.random.default_rng(0)
        steps = 0
        for seed, curric in [(1, 0.0), (2, 1.0), (3, 0.5), (4, 1.0), (5, 0.0)]:
            env.set_curriculum(curric)
            obs, _ = env.reset(seed=seed)
            for _ in range(400):
                self.assertEqual(obs[10], 0.0)
                self.assertEqual(obs[11], 0.0)
                self.assertLessEqual(abs(float(obs[8])), FW["ALT_OBS_CLAMP"] + 1e-7)
                obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 4))
                steps += 1
                if term or trunc:
                    obs, _ = env.reset()
        self.assertGreater(steps, 1000)

    def test_hvel_zero_while_really_moving(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options={**NOISELESS, "initial_velocity": [2, -1.5, 0, 0, 0, 0]})
        obs, *_ = env.step(np.zeros(4))
        self.assertGreater(np.linalg.norm(env._data.qvel[:2]), 1.0)
        np.testing.assert_array_equal(obs[10:12], 0)
        legacy = QuadrotorEnv(firmware_contract=False)
        legacy.reset(seed=0, options={**NOISELESS, "initial_velocity": [2, -1.5, 0, 0, 0, 0]})
        self.assertGreater(abs(legacy.step(np.zeros(4))[0][10]), 0.5)

    def test_obs8_obs9_follow_firmware_formula(self):
        env = QuadrotorEnv()
        for alt in (0.0, 0.25, 0.3, 0.33, 0.9):
            obs, _ = env.reset(seed=0, options={**NOISELESS, "initial_alt_agl": alt})
            for _ in range(7):
                obs, *_ = env.step(np.zeros(4))
            ctrl = FirmwareController({}, clamp_alt_obs=True)
            ref = ctrl.build_obs(np.zeros(3), np.zeros(3), 0, 0, env._meas_alt, env._meas_vvel)
            self.assertAlmostEqual(float(obs[8]), float(ref[8]), places=6)
            self.assertAlmostEqual(float(obs[9]), float(ref[9]), places=6)
        # On the ground (bench): obs[8] is pinned at -0.05 like harness TEST 7.
        obs, _ = env.reset(seed=0, options={**NOISELESS, "ground_start": True})
        self.assertAlmostEqual(float(obs[8]), -0.05, places=6)

    def test_fixed_target_ignores_altitude_setpoints(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options={**NOISELESS, "setpoint": {"alt_m": 1.0}})
        self.assertAlmostEqual(env.target_z, GROUND_REST_Z + 0.3)
        env.set_setpoint(alt_m=1.4)
        self.assertAlmostEqual(env.target_z, GROUND_REST_Z + 0.3)
        for _ in range(450):          # past every 200-step setpoint change
            env.step(np.zeros(4))
            self.assertAlmostEqual(env.target_z, GROUND_REST_Z + 0.3)
            self.assertEqual((env._cmd_roll, env._cmd_pitch), (0.0, 0.0))

    def test_reward_and_termination_use_firmware_target(self):
        env = QuadrotorEnv()
        level = {**NOISELESS, "initial_rpy_deg": [0, 0, 0]}
        env.reset(seed=0, options=level)
        self.assertAlmostEqual(env._data.qpos[2], GROUND_REST_Z + 0.3)
        # At the target, level, still: r_track = 1 → reward = 1 + 0.5 alive
        self.assertAlmostEqual(env._compute_reward(np.zeros(4)), 1.5, places=6)
        env.reset(seed=0, options={**level, "initial_alt_agl": 0.8})
        self.assertAlmostEqual(env._compute_reward(np.zeros(4)),
                               math.exp(-4 * 0.5 ** 2) + 0.5, places=6)


class YawSpinContract(unittest.TestCase):
    def test_contract_reward_penalises_steady_yaw_spin(self):
        def reward_at(fc, gz, curriculum=0.0):
            env = QuadrotorEnv(firmware_contract=fc)
            env.set_curriculum(curriculum)
            env.reset(seed=0, options={**NOISELESS, "initial_rpy_deg": [0, 0, 0]})
            env._data.qvel[5] = gz
            mujoco.mj_forward(env._model, env._data)
            return env._compute_reward(np.zeros(4))
        gz = math.radians(25)
        full = {"yaw_weight_curriculum": False}
        self.assertAlmostEqual(reward_at(full, 0) - reward_at(full, gz),
                               0.06 * gz ** 2 + (1 - math.exp(-3.0 * gz ** 2)), places=6)
        # ramped with curriculum: no yaw term at progress 0 (= the run that learned)
        self.assertAlmostEqual(reward_at(True, 0) - reward_at(True, gz), 0.06 * gz ** 2, places=6)
        self.assertAlmostEqual(reward_at(False, 0) - reward_at(False, gz), 0.06 * gz ** 2, places=6)

    def test_staying_alive_always_beats_crashing(self):
        """Regression for the 2026-11-21 collapse: with an unbounded yaw
        penalty, a fast-spinning drone earned < 0 per step, so crashing
        (−50 once) beat flying. Per-step reward must stay > 0 for any
        yaw rate when level, still and at the target."""
        env = QuadrotorEnv(firmware_contract={"yaw_weight_curriculum": False})
        env.reset(seed=0, options={**NOISELESS, "initial_rpy_deg": [0, 0, 0]})
        for gz in (0.0, 1.0, 5.0, 20.0, 34.0):           # rad/s, up to gyro full scale
            env._data.qvel[3:6] = [0, 0, gz]
            mujoco.mj_forward(env._model, env._data)
            r_track = env._compute_reward(np.zeros(4)) + 0.06 * gz ** 2 - 0.5   # strip legacy terms
            # the yaw term can never take more than ~2/3 of the tracking reward
            self.assertGreater(r_track, math.exp(-3.0 * 0.6 ** 2) - 1e-9, gz)

    def test_yaw_spin_gate(self):
        from evaluation import CONTRACT_SCENARIOS, score_trajectory
        hover = [s for s in CONTRACT_SCENARIOS if s.name == "hover30"][0]
        n = 3000
        t = np.arange(1, n + 1) * 0.01
        pos = np.zeros((n, 3)); pos[:, 2] = GROUND_REST_Z + 0.3
        for rate, ok in [(0.0, True), (20.0, False)]:
            rates = np.zeros((n, 3)); rates[:, 2] = rate
            r = score_trajectory(hover, t, pos, np.zeros((n, 3)), rates,
                                 np.full((n, 4), HOVER_THROTTLE), target_z=GROUND_REST_Z + 0.3)
            self.assertEqual(r["passed"], ok, r["failures"])
            self.assertEqual("yaw_spin" in r["failures"], not ok)


class AltitudeLoopContract(unittest.TestCase):
    def ctrl_after_step(self, overrides=None, **opts):
        env = QuadrotorEnv(firmware_contract=overrides or True)
        env.reset(seed=0, options={**NOISELESS, **opts})
        env.step(np.zeros(4))
        return env, env.last_motor_cmd.copy()

    def test_pd_term_changes_applied_thrust(self):
        for alt in (0.1, 0.5, 0.8, 2.0):
            env, ctrl = self.ctrl_after_step(initial_alt_agl=alt)
            adj = float(np.clip(FW["ALT_KP"] * (FW["TARGET_ALT"] - alt), -0.1, 0.1))
            np.testing.assert_allclose(ctrl, HOVER_THROTTLE + adj, atol=1e-12)
            self.assertAlmostEqual(env._alt_adj, adj, places=12)
            _, off = self.ctrl_after_step({"alt_kp": 0.0, "alt_kd": 0.0}, initial_alt_agl=alt)
            np.testing.assert_allclose(off, HOVER_THROTTLE, atol=1e-12)
            self.assertNotAlmostEqual(float(ctrl[0]), float(off[0]), places=4)

    def test_pd_term_changes_vertical_motion(self):
        def vz_after(overrides):
            env = QuadrotorEnv(firmware_contract=overrides)
            env.reset(seed=0, options={**NOISELESS, "initial_alt_agl": 0.6,
                                       "initial_rpy_deg": [0, 0, 0]})
            for _ in range(20):
                env.step(np.zeros(4))
            return env._data.qvel[2]
        # 0.3 m above target: adj = -0.045 → thrust drop through the measured
        # (nonlinear) thrust curve, held for 0.2 s
        drop = de.thrust_fraction(HOVER_THROTTLE - 0.045) - de.thrust_fraction(HOVER_THROTTLE)
        # Nominal actuator: the drop arrives after the dead time and (on
        # average) one lag time constant later, so it acts for 0.2 s minus both.
        acting = 0.2 - de.ACT_DELAY_TICKS_NOMINAL * 0.01 - de.ACT_TAU_NOMINAL
        expected = 4 * MAX_THRUST * drop / de.BASE_MASS * acting
        self.assertAlmostEqual(vz_after(True), expected, delta=0.1 * abs(expected))
        # HOVER_THROTTLE is rounded to 6 dp (it becomes a C float), so hover is
        # exact only to ~1e-6 thrust: allow 0.1 mm/s after 0.2 s.
        self.assertAlmostEqual(vz_after({"alt_kp": 0.0, "alt_kd": 0.0}), 0.0, delta=1e-4)

    def test_d_term_uses_baro_vvel(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options=NOISELESS)
        env._meas_alt, env._meas_vvel = 0.3, 1.5          # rising at target height
        env.step(np.zeros(4))
        self.assertAlmostEqual(env._alt_adj, -FW["ALT_KD"] * 1.5, places=12)

    def test_ground_start_ramp_matches_main_c(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options={**NOISELESS, "ground_start": True})
        a = np.array([0.5, -0.5, 0.2, -0.9])
        s = np.zeros(4)
        for k in range(1, FW["RAMP_TICKS"] + 3):
            s = 0.3 * a + 0.7 * s
            alt_seen, vvel_seen = env._meas_alt, env._meas_vvel   # sample the NN acted on
            _, _, term, _, info = env.step(a)
            self.assertFalse(term)
            ctrl = env.last_motor_cmd
            if k <= FW["RAMP_TICKS"]:
                f = k / FW["RAMP_TICKS"]
                np.testing.assert_allclose(
                    ctrl, np.clip(f * (HOVER_THROTTLE + ACTION_SCALE * s), f * 0.06, 1), atol=1e-6)
                self.assertTrue(info["ramping"])
            else:
                self.assertFalse(info["ramping"])
                adj = np.clip(0.15 * (0.3 - alt_seen) - 0.02 * vvel_seen, -0.1, 0.1)
                np.testing.assert_allclose(ctrl, np.clip(HOVER_THROTTLE + adj + ACTION_SCALE * s, 0, 1),
                                           atol=1e-6)

    def test_ground_start_lifts_off_and_weak_thrust_times_out(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options={**NOISELESS, "ground_start": True})
        for _ in range(150):
            _, _, term, _, info = env.step(np.zeros(4))
            self.assertFalse(term, info)        # resting below CRASH_ALT is not a crash
        self.assertTrue(env._airborne)          # nominal thrust + P term lifts it
        env.reset(seed=0, options={**NOISELESS, "ground_start": True, "thrust_scale": 0.6})
        for k in range(400):
            _, _, term, _, info = env.step(np.zeros(4))
            if term:
                break
        self.assertEqual(info["termination_reason"], "no_liftoff")
        self.assertEqual(k + 1, 300)

    def test_tilt_limit_is_firmware_60_degrees(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options={**NOISELESS, "initial_rpy_deg": [62, 0, 0], "initial_alt_agl": 1})
        self.assertEqual(env.step(np.zeros(4))[4]["termination_reason"], "tilt_limit")


class BarometerAndRandomisation(unittest.TestCase):
    def test_env_baro_equals_fw_common_baromodel(self):
        env = QuadrotorEnv()
        env.reset(seed=3, options={**NOISELESS, "initial_alt_agl": 0.7})
        ref = BaroModel(None, 0.0, init_alt=env._meas_alt, init_vvel=env._meas_vvel,
                        phase=env._baro_tick)
        for _ in range(120):                    # falls/climbs under the P+D loop
            env.step(np.zeros(4))
            ref.update(env._true_alt_agl()[0])     # same position sensor the env reads
            self.assertAlmostEqual(env._meas_alt, ref.alt, places=12)
            self.assertAlmostEqual(env._meas_vvel, ref.vvel, places=12)

    def test_baro_updates_at_25hz_with_lag(self):
        env = QuadrotorEnv()
        env.reset(seed=0, options={**NOISELESS, "initial_alt_agl": 0.6})
        alts = []
        for _ in range(40):
            env.step(np.zeros(4))
            alts.append(env._meas_alt)
        changes = np.flatnonzero(np.diff(alts))
        self.assertTrue(np.all(np.diff(changes) == 4))
        self.assertLess(abs(env._meas_vvel), abs(env._data.qvel[2]))   # EMA lags truth

    def test_baro_noise_level_and_seeded(self):
        env = QuadrotorEnv()
        _, info = env.reset(seed=0, options={"domain_randomization": False, "ground_start": True,
                                             "thrust_scale": 0.3})
        self.assertAlmostEqual(info["baro_noise_std"], de.BARO_NOISE_NOMINAL)
        alts, vv = [], []
        for _ in range(4000):
            env.step(np.full(4, -1.0))
            alts.append(env._meas_alt); vv.append(env._meas_vvel)
        alts = np.array(alts[::4])
        self.assertAlmostEqual(alts.std(), de.BARO_NOISE_NOMINAL, delta=0.002)
        # EMA(α) of differenced white noise: std = σ·α/T·sqrt(2/(2-α)) ≈ 0.038 m/s
        # (bench TEST 13 measured 0.036 m/s with alt σ 1.4 cm)
        pred = de.BARO_NOISE_NOMINAL * 0.1 / 0.04 * math.sqrt(2 / 1.9)
        self.assertAlmostEqual(np.std(vv), pred, delta=0.25 * pred)
        a = QuadrotorEnv(); b = QuadrotorEnv()
        oa, _ = a.reset(seed=9); ob, _ = b.reset(seed=9)
        for _ in range(50):
            np.testing.assert_array_equal(oa, ob)
            oa = a.step(np.full(4, 0.1))[0]; ob = b.step(np.full(4, 0.1))[0]

    def test_thrust_randomisation_covers_25_percent(self):
        env = QuadrotorEnv()
        scales, per_motor = [], []
        for seed in range(400):
            _, info = env.reset(seed=seed)
            scales.append(info["thrust_scale"])
            per_motor.append(env._max_thrust / (MAX_THRUST * info["thrust_scale"]))
        self.assertLess(min(scales), 0.77); self.assertGreater(max(scales), 1.23)
        self.assertTrue(0.75 <= min(scales) and max(scales) <= 1.25)
        per_motor = np.array(per_motor)
        self.assertTrue(np.all((per_motor >= 0.9) & (per_motor <= 1.1)))
        legacy = QuadrotorEnv(firmware_contract=False)
        legacy.reset(seed=1)
        self.assertTrue(np.all(np.abs(legacy._max_thrust / de.LEGACY_MAX_THRUST - 1) <= 0.1))

    def test_ground_rest_height(self):
        env = QuadrotorEnv(firmware_contract=False)
        env.reset(seed=0, options={**NOISELESS, "initial_position": [0, 0, 0.05]})
        env._data.ctrl[:] = 0
        for _ in range(4000):
            mujoco.mj_step(env._model, env._data)
        self.assertAlmostEqual(env._data.qpos[2], GROUND_REST_Z, delta=2e-4)
        self.assertLess(GROUND_REST_Z, de.CRASH_ALT)

    def test_config_switch(self):
        self.assertIsNone(QuadrotorEnv(firmware_contract=False).contract)
        self.assertIsNone(QuadrotorEnv(firmware_contract={"enabled": False}).contract)
        self.assertEqual(QuadrotorEnv().contract["thrust_scale_range"], (0.75, 1.25))
        self.assertEqual(QuadrotorEnv(firmware_contract={"baro_noise_std": 0.02}).contract["baro_noise_std"],
                         (0.02, 0.02))
        with self.assertRaises(ValueError):
            QuadrotorEnv(firmware_contract={"no_such_key": 1})
        import json
        json.dumps(QuadrotorEnv().contract_config())

    def test_check_env_both_modes(self):
        from stable_baselines3.common.env_checker import check_env
        for fc in (True, False):
            with self.subTest(firmware_contract=fc):
                check_env(QuadrotorEnv(firmware_contract=fc), warn=True)


class FirmwareParity(unittest.TestCase):
    def test_constants_match_firmware_sources(self):
        pairs = {
            ("fc_config.h", "TARGET_ALT"): de.FW_TARGET_ALT,
            ("fc_config.h", "ALT_SCALE"): de.ALT_SCALE,
            ("fc_config.h", "VVEL_SCALE"): de.VVEL_SCALE,
            ("fc_config.h", "ALT_OBS_CLAMP"): de.FW_ALT_OBS_CLAMP,
            ("fc_config.h", "ALT_KP"): de.FW_ALT_KP,
            ("fc_config.h", "ALT_KD"): de.FW_ALT_KD,
            ("fc_config.h", "ALT_ADJ_MAX"): de.FW_ALT_ADJ_MAX,
            ("fc_config.h", "ACTION_FILTER_ALPHA"): de.TRAIN_ACTION_ALPHA,
            ("baro.c", "VVEL_ALPHA"): de.FW_VVEL_ALPHA,
            ("baro.c", "BARO_DECIMATION"): de.FW_BARO_DECIMATION,
            ("main.c", "RAMP_TICKS"): de.FW_RAMP_TICKS,
            ("main.c", "TILT_LIMIT_DEG"): de.FW_TILT_LIMIT_DEG,
        }
        for (path, name), value in pairs.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(c_define(path, name), value, places=6)
        self.assertIn("ramp_frac * 0.06f", (REPO / "main.c").read_text())
        for key, value in dict(TARGET_ALT=de.FW_TARGET_ALT, ALT_OBS_CLAMP=de.FW_ALT_OBS_CLAMP,
                               ALT_KP=de.FW_ALT_KP, ALT_KD=de.FW_ALT_KD,
                               ALT_ADJ_MAX=de.FW_ALT_ADJ_MAX, VVEL_ALPHA=de.FW_VVEL_ALPHA,
                               BARO_DECIMATION=de.FW_BARO_DECIMATION,
                               RAMP_TICKS=de.FW_RAMP_TICKS, RAMP_FLOOR=de.FW_RAMP_FLOOR,
                               ACTION_SCALE=ACTION_SCALE).items():
            self.assertAlmostEqual(FW[key], value, places=6, msg=key)
        self.assertAlmostEqual(FW["HOVER_THROTTLE"], HOVER_THROTTLE, places=4)
        d = FIRMWARE_CONTRACT_DEFAULTS
        self.assertEqual((d["alt_kp"], d["alt_kd"], d["alt_adj_max"], d["target_alt_agl"]),
                         (FW["ALT_KP"], FW["ALT_KD"], FW["ALT_ADJ_MAX"], FW["TARGET_ALT"]))

    def test_firmware_controller_ramp(self):
        w = dict(W1=np.zeros((32, 12), np.float32), B1=np.zeros(32, np.float32),
                 W2=np.zeros((32, 32), np.float32), B2=np.zeros(32, np.float32),
                 W3=np.zeros((4, 32), np.float32), B3=np.float32([0.5, -0.5, 0, 0]))
        c = FirmwareController(w, ramp_ticks=50)
        s = np.zeros(4)
        for k in range(1, 53):
            m, dbg = c.step(np.zeros(3), [0, 0, 1], 0, 0, 0.0, 0.0)
            s = 0.3 * np.array([0.5, -0.5, 0, 0]) + 0.7 * s
            if k <= 50:
                f = k / 50
                np.testing.assert_allclose(m, np.clip(f * (FW["HOVER_THROTTLE"] + 0.4 * s), f * .06, 1), atol=1e-6)
            else:
                np.testing.assert_allclose(m, FW["HOVER_THROTTLE"] + 0.045 + 0.4 * s, atol=1e-6)


class ActuatorDelayContract(unittest.TestCase):
    def test_actuator_model_dead_time_and_lag(self):
        a = de.ActuatorModel(3, 0.0, np.zeros(4))
        outs = [a.step(np.ones(4))[0] for _ in range(5)]
        np.testing.assert_allclose(outs, [0, 0, 0, 1, 1])
        b = de.ActuatorModel(0, 0.01, np.zeros(4))
        first = b.step(np.ones(4))[0]
        # exact tick average of 1 - exp(-t/tau) over 10 ms with tau = 10 ms
        self.assertAlmostEqual(first, 1 - (1 - np.exp(-1.0)), places=9)
        self.assertAlmostEqual(b.state[0], 1 - np.exp(-1.0), places=9)

    def test_randomisation_and_overrides(self):
        env = QuadrotorEnv()
        env.set_curriculum(1.0)                  # delay is phased in with progress
        delays, taus, inert = set(), [], []
        for seed in range(200):
            _, info = env.reset(seed=seed)
            delays.add(info["actuator_delay_ticks"]); taus.append(info["motor_tau_s"])
            inert.append(info["inertia_scale"])
        self.assertEqual(delays, {2, 3, 4, 5})
        self.assertTrue(0.005 <= min(taus) and max(taus) <= 0.03)
        self.assertTrue(0.8 <= min(inert) and max(inert) <= 1.5)
        _, info = env.reset(seed=0, options={**NOISELESS})
        self.assertEqual((info["actuator_delay_ticks"], info["motor_tau_s"], info["inertia_scale"]),
                         (de.ACT_DELAY_TICKS_NOMINAL, de.ACT_TAU_NOMINAL, 1.0))
        _, info = env.reset(seed=0, options={**NOISELESS, "actuator_delay_ticks": 0,
                                             "motor_tau_s": 0.0, "inertia_scale": 1.3})
        self.assertEqual(info["actuator_delay_ticks"], 0)
        np.testing.assert_allclose(env._model.body_inertia[env._drone_body_id],
                                   env._nominal_inertia * 1.3)

    def test_delay_phased_in_with_curriculum(self):
        env = QuadrotorEnv()
        env.set_curriculum(0.0)
        infos = [env.reset(seed=s)[1] for s in range(50)]
        self.assertTrue(all(i["actuator_delay_ticks"] == 0 and i["motor_tau_s"] == 0 for i in infos))
        full = QuadrotorEnv(firmware_contract={"actuator_curriculum": False})
        full.set_curriculum(0.0)
        self.assertEqual({full.reset(seed=s)[1]["actuator_delay_ticks"] for s in range(200)}, {2, 3, 4, 5})

    def test_hover_holds_with_delay(self):
        env = QuadrotorEnv(firmware_contract={"alt_kp": 0.0, "alt_kd": 0.0})
        env.reset(seed=0, options={**NOISELESS, "initial_alt_agl": 0.5, "initial_rpy_deg": [0, 0, 0]})
        for _ in range(100):
            env.step(np.zeros(4))
        self.assertLess(abs(env._data.qvel[2]), 1e-3)   # air start: motors already at hover


class LegacyReproduction(unittest.TestCase):
    def test_legacy_mode_bit_identical_to_pre_contract_env(self):
        golden = np.load(HERE / "fixtures" / "legacy_obs_golden.npz")
        now = record(lambda: QuadrotorEnv(firmware_contract=False))
        for key in golden.files:
            with self.subTest(key=key):
                np.testing.assert_array_equal(now[key], golden[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
