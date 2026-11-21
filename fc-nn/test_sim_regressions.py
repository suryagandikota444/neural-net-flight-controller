"""Regression tests for fixes 4–6: real physics, complete resets, honest metrics."""
import unittest
import numpy as np
import mujoco
from drone_env import QuadrotorEnv, BASE_MASS, GRAVITY, HOVER_THROTTLE, _quat_to_euler
from test_env_physics import raw_step
from evaluation import Scenario, score_trajectory, first_sustained_time
from validate import pwm_from_output

FIXED = dict(domain_randomization=False, randomize_setpoints=False,
             setpoint={"alt_m": 1.0}, initial_rpy_deg=[0,0,0], sensor_noise=False)


class PhysicsAndResetTests(unittest.TestCase):
    def setUp(self):
        self.e = QuadrotorEnv()
        self.e.reset(seed=42, options=FIXED)

    def tearDown(self):
        self.e.close()

    def test_mass_and_inertia_are_whole_aircraft(self):
        e = self.e
        self.assertAlmostEqual(e._model.body_subtreemass[e._drone_body_id], BASE_MASS, places=12)
        np.testing.assert_allclose(e._model.body_inertia[e._drone_body_id], [.018,.018,.035])
        raw_step(e, np.full(4, HOVER_THROTTLE), 200)
        self.assertAlmostEqual(e._data.qpos[2], 1, places=8)
        self.assertAlmostEqual(e._get_obs()[5], .5, places=6)

    def test_free_fall(self):
        raw_step(self.e, np.zeros(4), 20)
        self.assertAlmostEqual(self.e._data.qpos[2], 1-.5*GRAVITY*.2**2, places=8)
        self.assertLess(np.linalg.norm(self.e._sensor(self.e._accel_sensor_id,3)), 1e-8)

    def test_cog_changes_actual_inertial_position_and_torque(self):
        e = self.e
        for sign in [-1,1]:
            offset = sign*np.array([.002,.001,0])
            e.reset(seed=42, options={**FIXED, "cog_offset": offset})
            np.testing.assert_allclose(e._data.xipos[e._drone_body_id]-e._data.xpos[e._drone_body_id], offset, atol=1e-12)
            e._data.ctrl[e._act_ids] = HOVER_THROTTLE
            mujoco.mj_forward(e._model,e._data)
            expected = np.cross(-offset,[0,0,BASE_MASS*GRAVITY]) / e._model.body_inertia[e._drone_body_id]
            np.testing.assert_allclose(e._data.qacc[3:6], expected, atol=1e-9)
        e.reset(seed=42, options=FIXED)
        np.testing.assert_allclose(e._data.xipos[e._drone_body_id]-e._data.xpos[e._drone_body_id],0,atol=1e-12)

    def test_mass_inertia_scaling_does_not_accumulate(self):
        for factor in [1.05,.95,1.0,1.05]:
            self.e.reset(seed=42, options={**FIXED,"mass":BASE_MASS*factor})
            np.testing.assert_allclose(self.e._model.body_inertia[self.e._drone_body_id],np.array([.018,.018,.035])*factor)
            self.assertAlmostEqual(self.e._model.body_subtreemass[self.e._drone_body_id],BASE_MASS*factor)

    def test_combined_initial_angles_and_rotated_cog(self):
        e=self.e
        offset=np.array([.002,.001,0])
        e.reset(seed=42,options={**FIXED,"initial_rpy_deg":[20,15,10],"cog_offset":offset})
        np.testing.assert_allclose(np.degrees(_quat_to_euler(e._data.qpos[3:7])),[20,15,10],atol=1e-10)
        rotation=e._data.xmat[e._drone_body_id].reshape(3,3)
        np.testing.assert_allclose(e._data.xipos[e._drone_body_id]-e._data.xpos[e._drone_body_id],rotation@offset,atol=1e-12)

    def test_complete_seeded_reset_and_fixed_commands(self):
        e=self.e
        options={**FIXED, "sensor_noise":True, "max_steps":300}
        initial,_=e.reset(seed=123,options=options)
        expected=[e.step([.1,-.1,.2,-.2])[0] for _ in range(10)]
        e.set_setpoint(roll_deg=20,pitch_deg=15,alt_m=.4)
        actual,_=e.reset(seed=123,options=options)
        np.testing.assert_array_equal(actual,initial)
        np.testing.assert_array_equal(e._smooth_action,0)
        for obs in expected: np.testing.assert_array_equal(e.step([.1,-.1,.2,-.2])[0],obs)
        e.reset(seed=123,options=options)
        for _ in range(250): e.step(np.zeros(4))
        self.assertEqual((e._cmd_roll,e._cmd_pitch,e._cmd_alt),(0,0,1))

    def test_invalid_action_rejected_before_physics(self):
        for action in ([0,0,0,float('nan')],[0,0,0]):
            with self.assertRaises(ValueError): self.e.step(action)
            self.assertEqual(self.e._data.time,0)

    def test_residual_pwm_mapping(self):
        np.testing.assert_allclose(pwm_from_output(np.zeros(4)),1000+1000*HOVER_THROTTLE)
        self.assertEqual(pwm_from_output(np.full(4,-10))[0],1000)
        np.testing.assert_allclose(pwm_from_output(np.full(4,10)),1000+1000*(HOVER_THROTTLE+.4))


class MetricTests(unittest.TestCase):
    def score(self, *, seconds=1, length=100, terminated=False, drift=0):
        times=np.arange(1,length+1)*.01
        pos=np.zeros((length,3));pos[:,2]=1;pos[:,0]=drift
        return score_trajectory(Scenario('test',seconds),times,pos,np.zeros((length,3)),
            np.zeros((length,3)),np.full((length,4),HOVER_THROTTLE),terminated=terminated)

    def test_single_crossing_is_not_recovery(self):
        times=np.arange(1,301)*.01
        band=np.zeros(300,dtype=bool);band[10]=True
        self.assertIsNone(first_sustained_time(times,band,1))
        band[100:200]=True
        self.assertAlmostEqual(first_sustained_time(times,band,1),1.01)

    def test_perfect_full_hover_passes(self): self.assertTrue(self.score()['passed'])
    def test_short_perfect_trajectory_fails(self): self.assertFalse(self.score(length=20)['passed'])
    def test_crash_fails_despite_zero_rmse(self): self.assertFalse(self.score(terminated=True)['passed'])
    def test_horizontal_envelope_is_checked(self): self.assertIn('horizontal_envelope',self.score(drift=3)['failures'])


if __name__=='__main__': unittest.main(verbosity=2)
