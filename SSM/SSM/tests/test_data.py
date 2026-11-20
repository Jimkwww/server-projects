import json
import numpy as np
import pytest
import h5py
from ssm_exp.data import read_manifest,audit,IQDataset

@pytest.mark.parametrize('fmt',['h5','npy'])
def test_manifest_and_overlap(tmp_path,fmt):
    x=np.arange(12*16*2,dtype=np.float32).reshape(12,16,2);y=np.arange(12)%3;z=np.arange(12)-6
    m={'format':fmt,'layout':'NLC','label_format':'integer','normalization':'none','classes':3,
       'splits':{'train':'train.npy','val':'val.npy'}}
    if fmt=='h5':
        with h5py.File(tmp_path/'data.h5','w') as f:
            f['X']=x;f['Y']=y;f['Z']=z
        m['file']='data.h5'
    else:
        for k,v in [('x',x),('y',y),('snr',z)]:np.save(tmp_path/(k+'.npy'),v);m[k]=k+'.npy'
    np.save(tmp_path/'train.npy',np.arange(8));np.save(tmp_path/'val.npy',np.arange(8,12))
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    manifest=read_manifest(tmp_path/'manifest.json');audit(manifest)
    sample=IQDataset(manifest,'val')[0]
    np.testing.assert_array_equal(sample[0].numpy(),x[8].T)
    assert sample[1:]==(2,2.,8)
    np.save(tmp_path/'val.npy',np.arange(7,12))
    with pytest.raises(ValueError,match='overlap'):audit(manifest)
