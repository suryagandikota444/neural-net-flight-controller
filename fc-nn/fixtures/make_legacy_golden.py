"""Record observations/rewards from the pre-contract drone_env.py (git 4e16bf5 +
working tree of 2026-11-21) so tests can prove firmware_contract=False still
reproduces the old environment bit-for-bit. Regenerate ONLY from the old code."""
import inspect
import numpy as np
from drone_env import QuadrotorEnv

def record(make_env):
    out = {}
    act_rng = np.random.default_rng(2024)
    actions = act_rng.uniform(-0.6, 0.6, size=(450, 4)).astype(np.float32)
    for name, seed, options, curric in [
            ("train_default", 7, None, 0.0),
            ("train_curric", 11, None, 0.7),
            ("fixed", 123, dict(domain_randomization=False, randomize_setpoints=False,
                                setpoint={"alt_m": 1.0}, initial_rpy_deg=[5, -3, 0]), 0.0)]:
        env = make_env()
        env.set_curriculum(curric)
        obs, _ = env.reset(seed=seed, options=options)
        obs_l, rew_l, term_l = [obs], [], []
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); rew_l.append(r); term_l.append(term)
            if term or trunc:
                obs, _ = env.reset()
                obs_l.append(obs)
        out[name + "_obs"] = np.array(obs_l)
        out[name + "_rew"] = np.array(rew_l)
        out[name + "_term"] = np.array(term_l)
        env.close()
    out["actions"] = actions
    return out

if __name__ == "__main__":
    import pathlib
    if "firmware_contract" in inspect.signature(QuadrotorEnv.__init__).parameters:
        raise SystemExit("refusing: run against the pre-contract drone_env.py only")
    np.savez_compressed(pathlib.Path(__file__).with_name("legacy_obs_golden.npz"),
                        **record(QuadrotorEnv))
    print("wrote legacy_obs_golden.npz")
