import pyarrow.parquet as pq
import numpy as np

pf = pq.ParquetFile('/data1/user_wzk/datasets/EM-Radar-mixed/BSS_train_0000.parquet')
tbl = pf.read_row_group(0)
df = tbl.to_pandas()

def stack_col(col):
    # each row is a (1024,) object array of (2,) float arrays -> [N,1024,2]
    return np.array([np.stack(row) for row in col.values], dtype=np.float64)

iq = stack_col(df['iq'])
s1 = stack_col(df['source_1'])
s2 = stack_col(df['source_2'])
print('shapes:', iq.shape, s1.shape, s2.shape, iq.dtype)

# additivity: iq == source_1 + source_2 ?
resid = iq - (s1 + s2)
print('additivity |iq-(s1+s2)|: max=%.3e mean=%.3e' % (np.abs(resid).max(), np.abs(resid).mean()))

# time-alignment: cross-correlation at lag 0 vs shifted
def normx(x):
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
c0 = (normx(iq) * normx(s1)).sum((1, 2))
s1_shift = np.roll(s1, 1, axis=1)
c1 = (normx(iq) * normx(s1_shift)).sum((1, 2))
print('corr(iq,s1) lag0 mean=%.4f  lag1 mean=%.4f  (lag0 >> lag1 => 时间对齐)' % (c0.mean(), c1.mean()))

# signal power / normalization stats
print('iq   |.| max=%.3f mean=%.3f' % (np.abs(iq).max(), np.abs(iq).mean()))
print('s1   |.| max=%.3f mean=%.3f' % (np.abs(s1).max(), np.abs(s1).mean()))
print('s2   |.| max=%.3f mean=%.3f' % (np.abs(s2).max(), np.abs(s2).mean()))

# which labels are pure (single source) vs mixed
labels = df['label'].values
single = [l for l in np.unique(labels) if '+' not in l]
print('\nsingle-source labels (%d):' % len(single), sorted(single))
print('num mixed labels:', sum(1 for l in np.unique(labels) if '+' in l))
