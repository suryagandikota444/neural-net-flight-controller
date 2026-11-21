"""Read float32 actor tensors without importing the cloud-backed Torch install."""
import io, pickle, zipfile, collections
import numpy as np

def load_sb3_weights(path):
    with zipfile.ZipFile(path) as outer:
        raw=outer.read('policy.pth')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert archive.read('archive/byteorder')==b'little'
        def rebuild(storage,offset,size,stride,*args):
            arr=np.ndarray(shape=size,dtype='<f4',buffer=storage,offset=offset*4,
                strides=tuple(s*4 for s in stride))
            return arr.copy()
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('collections','OrderedDict'): return collections.OrderedDict
                if (module,name)==('torch','FloatStorage'): return 'float32'
                if (module,name)==('torch._utils','_rebuild_tensor_v2'): return rebuild
                raise ValueError((module,name))
            def persistent_load(self,pid):
                kind,dtype,key,device,count=pid
                assert kind=='storage' and dtype=='float32'
                b=archive.read('archive/data/'+key)
                assert len(b)==count*4
                return b
        d=Reader(io.BytesIO(archive.read('archive/data.pkl'))).load()
    names={'W1':'mlp_extractor.policy_net.0.weight','B1':'mlp_extractor.policy_net.0.bias',
           'W2':'mlp_extractor.policy_net.2.weight','B2':'mlp_extractor.policy_net.2.bias',
           'W3':'action_net.weight','B3':'action_net.bias'}
    return {k:d[n] for k,n in names.items()}
