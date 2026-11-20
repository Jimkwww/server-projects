import pyarrow.parquet as pq
import numpy as np

pf = pq.ParquetFile('/data1/user_wzk/datasets/EM-Radar-mixed/BSS_train_0000.parquet')
print('schema:')
print(pf.schema)
print('num_rows:', pf.metadata.num_rows)

# read a small sample of rows to inspect values
tbl = pf.read_row_group(0)
df = tbl.slice(0, 5).to_pandas()
print('columns:', list(df.columns))
for c in df.columns:
    v = df[c].iloc[0]
    print(f'\n--- col {c} ---')
    print('  dtype:', df[c].dtype)
    print('  sample value (repr, truncated):', repr(v)[:400])
