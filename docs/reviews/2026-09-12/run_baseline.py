"""Reproducible review baseline, preserving the repository's original physics.

Runs actor tensors directly through fw_common.nn_forward_np to avoid requiring
Torch/SB3 imports. This is a diagnostic benchmark, not a deployment gate.
Fixed setpoints, fresh environments/filter state, fixed seeds 100..104.
Outputs do not certify corrected physics or the firmware control path.
"""
import sys, json, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'fc-nn'))
import numpy as np
import mujoco
from fw_common import nn_forward_np
from checkpoint_arrays import load_sb3_weights
from drone_env import QuadrotorEnv, BASE_MASS, MAX_THRUST, KM_KF_RATIO, _quat_to_euler

def episode(weights, seed, seconds, tilt=0, wind=0):
    e=QuadrotorEnv()
    e._randomize_setpoints=False
    e.set_setpoint(alt_m=1.0)
    e._max_ep_steps=round(seconds*100)
    e.reset(seed=seed)
    e._max_thrust[:]=MAX_THRUST
    e._km_kf[:]=KM_KF_RATIO
    e._wind_force[:]=[wind,0,0]
    e._model.body_mass[e._drone_body_id]=BASE_MASS
    e._model.body_ipos[e._drone_body_id]=0
    e._data.qpos[:3]=[0,0,1]
    a=math.radians(tilt)/2
    e._data.qpos[3:7]=[math.cos(a),math.sin(a),0,0]
    e._data.qvel[:]=0
    mujoco.mj_forward(e._model,e._data)
    obs=e._get_obs()
    zs=[]; angles=[]; xys=[]; recovery=None; crashed=False
    for i in range(e._max_ep_steps):
        obs,_,term,trunc,_=e.step(nn_forward_np(weights,obs))
        r,p,_=_quat_to_euler(e._data.qpos[3:7])
        zs.append(float(e._data.qpos[2]))
        angles.append([math.degrees(r),math.degrees(p)])
        xys.append(float(np.linalg.norm(e._data.qpos[:2])))
        if recovery is None and max(abs(angles[-1][0]),abs(angles[-1][1]))<10:
            recovery=(i+1)*0.01
        if term or trunc:
            crashed=bool(term)
            break
    e.close()
    z=np.asarray(zs); att=np.asarray(angles)
    return dict(seed=seed, crashed=crashed, seconds=len(z)*.01,
        alt_rmse=float(np.sqrt(np.mean((z-1)**2))),
        within5cm=float(np.mean(abs(z-1)<.05)), within10cm=float(np.mean(abs(z-1)<.1)),
        att_rms=float(np.sqrt(np.mean(np.sum(att**2,axis=1)))),
        max_drift=max(xys), recovery_s=recovery)

import argparse
parser=argparse.ArgumentParser(description='Repeat the September 12 review baseline: 4 checkpoints, 3 scenarios, 5 seeds. Does not alter simulation or model files.')
parser.add_argument('--out', type=Path, default=Path(__file__).with_name('policy-results-latest.json'))
args=parser.parse_args()
root=REPO/'fc-nn'
results={}
for model in sorted((root/'runs').glob('*/*.zip')):
    w=load_sb3_weights(model)
    key=str(model.relative_to(root))
    results[key]={}
    for name,kw in [('hover30',dict(seconds=30)),('tilt30deg',dict(seconds=2,tilt=30)),('wind10',dict(seconds=10,wind=.5))]:
        runs=[episode(w,seed,**kw) for seed in range(100,105)]
        results[key][name]=runs
        print(key,name,'survived',sum(not r['crashed'] for r in runs),'/5',
            'alt_rmse',round(np.mean([r['alt_rmse'] for r in runs]),3),
            'within5cm',round(np.mean([r['within5cm'] for r in runs])*100,1),
            'within10cm',round(np.mean([r['within10cm'] for r in runs])*100,1),
            'att_rms',round(np.mean([r['att_rms'] for r in runs]),2),
            'max_drift',round(max(r['max_drift'] for r in runs),2),
            'recovery',[r['recovery_s'] for r in runs] if 'tilt' in name else '',flush=True)
    args.out.write_text(json.dumps(results,indent=2))
