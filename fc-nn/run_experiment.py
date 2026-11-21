"""Controlled plant ablations and a seeded PPO warm-start; leaves flight weights intact."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import mujoco
from drone_env import QuadrotorEnv, BASE_MASS, MAX_THRUST, KM_KF_RATIO
from evaluation import evaluate, summarize, SCENARIOS
from fw_common import load_sb3_weights, nn_forward_np


def plant_factory(variant, legacy_source=None):
    if variant == "corrected": return QuadrotorEnv
    if variant == "mass_only":
        class NoCogEnv(QuadrotorEnv):
            def reset(self, *, seed=None, options=None):
                return super().reset(seed=seed, options={**(options or {}), "cog_offset": [0,0,0]})
        return NoCogEnv
    if not legacy_source: raise ValueError("legacy plant requires --legacy-source")
    spec = importlib.util.spec_from_file_location("legacy_drone_env", Path(legacy_source)/"drone_env.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    class LegacyEnv(legacy.QuadrotorEnv):
        """Original dynamics/controller, with the SAME deterministic test contract."""
        def _get_obs(self):
            if getattr(self,"_defer_obs",False): return np.zeros(12,dtype=np.float32)
            return super()._get_obs()

        def reset(self, *, seed=None, options=None):
            options = options or {}
            self._randomize_setpoints = False
            self.set_setpoint(**options["setpoint"])
            self._max_ep_steps = options["max_steps"]
            self._defer_obs = True
            super().reset(seed=seed)
            self._defer_obs = False
            if not options["domain_randomization"]:
                self._max_thrust = np.full(4,MAX_THRUST)
                self._km_kf = np.full(4,KM_KF_RATIO)
                self._model.body_mass[self._drone_body_id] = BASE_MASS
                self._model.body_ipos[self._drone_body_id] = 0
                self._wind_force = np.asarray(options["wind_force"],dtype=float)
            q = np.zeros(4)
            mujoco.mju_euler2Quat(q,np.radians(options["initial_rpy_deg"]),"XYZ")
            self._data.qpos[:3] = [0,0,1]
            self._data.qpos[3:7] = q
            self._data.qvel[:] = 0
            for i,aid in enumerate(self._act_ids): self._model.actuator_gear[aid,2]=self._max_thrust[i]
            # Deliberately retain old derived-state/same-frame behavior.
            mujoco.mj_forward(self._model,self._data)
            mass = self._model.body_mass[self._drone_body_id] + .220
            return self._get_obs(), {"mass_kg":float(mass),
                "cog_offset_m":self._data.xipos[self._drone_body_id]-self._data.xpos[self._drone_body_id]}
    return LegacyEnv


def weights_from_model(model):
    layers = [layer for layer in model.policy.mlp_extractor.policy_net if hasattr(layer,"weight")]
    layers.append(model.policy.action_net)
    return {f"{name}{i}":getattr(layer,attr).detach().cpu().numpy().astype(np.float32)
            for i,layer in enumerate(layers,1) for name,attr in [("W","weight"),("B","bias")]}


def selection_score(results):
    # Acceptance dominates completion; error breaks ties. Never discard crashes.
    return (sum(r["passed"] for r in results), sum(r["completed"] for r in results),
            -sum(r.get("altitude_rmse_m",100)+.01*r.get("attitude_rms_deg",1000) for r in results))


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from train import PPO_KWARGS
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    root=Path(__file__).resolve().parent.parent
    package_info=subprocess.check_output([os.sys.executable,"-m","pip","freeze"],text=True) if importlib.util.find_spec('pip') else subprocess.check_output(['uv','pip','freeze','--python',os.sys.executable],text=True)
    (out/'environment.txt').write_text(package_info)
    manifest=dict(seed=args.seed,requested_timesteps=args.timesteps,n_envs=args.n_envs,
                  fixed_curriculum=0.0,task="original training task at fixed easy curriculum",
                  warm_start=str(Path(args.model).resolve()),
                  model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [root/'fc-nn'/n for n in ('drone_env.py','quadrotor.xml','evaluation.py','train.py','run_experiment.py')]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'source.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=root))
    # Fixed curriculum isolates physics corrections; no implicit callback resume.
    env=make_vec_env(QuadrotorEnv,n_envs=args.n_envs,seed=args.seed,vec_env_cls=SubprocVecEnv)
    kwargs={**PPO_KWARGS,"tensorboard_log":str(out/'tensorboard'),"seed":args.seed,"device":"cpu"}
    model=PPO.load(args.model,env=env,**kwargs)
    initial_steps=model.num_timesteps

    class BenchmarkCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval=250000
            self.best_score=None
            self.records=[]

        def benchmark(self,label):
            weights=weights_from_model(self.model)
            results=evaluate(lambda obs:nn_forward_np(weights,obs),
                             seeds=range(100,105),output=out/f'eval_{label}.json')
            score=selection_score(results)
            record=dict(label=label,additional_timesteps=self.model.num_timesteps-initial_steps,
                        score=score,elapsed_s=time.monotonic()-start,summary=summarize(results))
            self.records.append(record)
            if self.best_score is None or score>self.best_score:
                self.best_score=score
                self.model.save(out/'best_model')
                (out/'best_selection.json').write_text(json.dumps(record,indent=2))
            (out/'progress.json').write_text(json.dumps(self.records,indent=2))
            print('EXPERIMENT_EVAL',json.dumps({k:v for k,v in record.items() if k!='summary'}),flush=True)

        def _on_training_start(self): self.benchmark('initial')
        def _on_step(self): return True
        def _on_rollout_end(self):
            additional=self.model.num_timesteps-initial_steps
            if additional>=self.next_eval:
                self.model.save(out/f'checkpoint_{additional}')
                self.benchmark(str(additional))
                self.next_eval+=250000

    callback=BenchmarkCallback()
    try:
        model.learn(total_timesteps=args.timesteps,callback=callback,reset_num_timesteps=False)
        model.save(out/'final_model')
        callback.benchmark('final')
        manifest.update(actual_additional_timesteps=model.num_timesteps-initial_steps,
                        elapsed_s=time.monotonic()-start,status='completed')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['benchmark','train'])
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--plant',choices=['legacy','mass_only','corrected'],default='corrected')
    p.add_argument('--legacy-source')
    p.add_argument('--seed-start',type=int,default=100)
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--trajectories')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--timesteps',type=int,default=1000000)
    p.add_argument('--n-envs',type=int,default=4)
    args=p.parse_args()
    if args.runs<1 or args.timesteps<1 or args.n_envs<1:p.error('counts must be positive')
    if args.mode=='train':return train(args)
    weights=load_sb3_weights(args.model)
    results=evaluate(lambda obs:nn_forward_np(weights,obs),seeds=range(args.seed_start,args.seed_start+args.runs),
        env_factory=plant_factory(args.plant,args.legacy_source),output=args.out,trajectories=args.trajectories)
    payload=json.loads(Path(args.out).read_text())
    payload['experiment']={'plant':args.plant,'model':str(Path(args.model).resolve()),
        'model_sha256':hashlib.sha256(Path(args.model).read_bytes()).hexdigest()}
    Path(args.out).write_text(json.dumps(payload,indent=2,allow_nan=False))


if __name__=='__main__':main()
