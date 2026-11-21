"""Compile actual nn.c against a candidate in a temporary directory, preserving weights.h."""
from pathlib import Path
import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'fc-nn'))
import numpy as np
from fw_common import load_sb3_weights, nn_forward_np
from export_weights import write_header
from test_nn_parity import golden_observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    weights=load_sb3_weights(args.model)
    observations=[obs for _,obs in golden_observations()]
    observations.extend(np.random.default_rng(987).uniform(-1,1,(1000,12)).astype(np.float32))
    with tempfile.TemporaryDirectory(prefix='fc-candidate-parity-') as dirname:
        folder=Path(dirname)
        for name in ('nn.c','nn.h'):shutil.copy2(ROOT/name,folder/name)
        write_header(*[weights[k] for k in ('W1','B1','W2','B2','W3','B3')],str(folder/'weights.h'))
        subprocess.run(['cc','-shared','-fPIC','-O1','-Wno-missing-braces',str(folder/'nn.c'),'-o',str(folder/'libnn.so')],check=True)
        lib=ctypes.CDLL(str(folder/'libnn.so'))
        ptr=ctypes.POINTER(ctypes.c_float)
        lib.nn_forward.argtypes=[ptr,ptr]
        lib.nn_forward.restype=None
        worst=0.0
        for obs in observations:
            obs=np.ascontiguousarray(obs,dtype=np.float32)
            actual=np.zeros(4,dtype=np.float32)
            lib.nn_forward(obs.ctypes.data_as(ptr),actual.ctypes.data_as(ptr))
            expected=nn_forward_np(weights,obs)
            assert np.all(np.isfinite(actual))
            worst=max(worst,float(np.max(np.abs(actual-expected))))
        result=dict(model=str(Path(args.model).resolve()),observations=len(observations),
                    max_absolute_error=worst,tolerance=1e-5,passed=worst<1e-5)
        Path(args.out).write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
