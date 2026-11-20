import sys
import numpy as np
import pyarrow.parquet as pq

parquet_path = sys.argv[1]
out_npy = sys.argv[2]
row_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

pf = pq.ParquetFile(parquet_path)
tbl = pf.read()
iq = np.asarray(tbl.column("iq")[row_idx].as_py(), dtype=np.float32)
label = tbl.column("label_value")[row_idx].as_py()
sample_rate = float(tbl.column("sample_rate")[row_idx].as_py())
sample_id = tbl.column("sample_id")[row_idx].as_py()

np.save(out_npy, iq)
print(f"sample_id={sample_id}")
print(f"label_value={label}")
print(f"sample_rate={sample_rate}")
print(f"iq_shape={iq.shape}  dtype={iq.dtype}")
print(f"saved -> {out_npy}")
