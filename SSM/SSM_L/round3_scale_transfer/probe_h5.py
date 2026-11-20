import h5py
import json

p = json.load(open('/home/user_wzk/wzk/projects/SSM/SSM_L/configs/data.json'))['file']
f = h5py.File(p, 'r')
print('keys', list(f.keys()))
print('X', f['X'].shape, 'Y', f['Y'].shape, 'Z', f['Z'].shape)
