import sys
import pyarrow.parquet as pq

path = sys.argv[1]
pf = pq.ParquetFile(path)
print("schema names:", pf.schema_arrow.names)
print("schema types:", [(n, str(t)) for n, t in zip(pf.schema_arrow.names, pf.schema_arrow.types)])
print("num_rows:", pf.metadata.num_rows)
print("num_row_groups:", pf.metadata.num_row_groups)
tbl = pf.read_row_group(0)
print("=== first 2 rows ===")
for name in tbl.column_names:
    col = tbl.column(name)
    print(f"--- {name} (type {col.type}) ---")
    print(col.slice(0, 2).to_pylist())
