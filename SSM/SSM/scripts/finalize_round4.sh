#!/usr/bin/env bash
# Round-4 finalize: wait for C_bi_front, then eval + benchmark it, merge all eval/bench
# results, and run aggregation. Runs detached in tmux (launch_tmux_round4.sh already ran
# training; this is the post-training mechanical pipeline).
set -euo pipefail
cd /home/user_wzk/wzk/projects/SSM/SSM
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh
conda activate SFM
OUT=runs/round4_structure
RPT=reports/round4_structure
mkdir -p "$RPT"

for s in 17 42 2026; do
  while [ ! -f "$OUT/C_bi_front_seed$s/DONE.json" ]; do sleep 30; done
done
echo "C_bi_front all done ($(date '+%H:%M:%S'))"

python scripts/eval_round4.py \
  --runs "$OUT/C_bi_front_seed17,$OUT/C_bi_front_seed42,$OUT/C_bi_front_seed2026" \
  --manifest configs/data_raw.json --device cuda:0 --pred-dir predictions_round4_structure \
  --out "$RPT/eval_cases_cbi.jsonl"

python scripts/bench_round4.py --runs "$OUT/C_bi_front_seed17" --device cuda:0 \
  --out "$RPT/benchmark_cbi.json"

python - <<'PY'
import json
from pathlib import Path
R = Path('reports/round4_structure')
order = ['G', 'G_corr', 'S1', 'A64', 'A128', 'B_front', 'C_bi_front', 'D_complex']
items = {}
for f in ['benchmark_partial.json', 'benchmark_cbi.json']:
    for b in json.loads((R / f).read_text()):
        items[b['variant']] = b
(R / 'benchmark.json').write_text(json.dumps([items[v] for v in order if v in items], indent=2))
PY

cat "$RPT/eval_cases_baseline.jsonl" "$RPT/eval_cases_r4.jsonl" "$RPT/eval_cases_cbi.jsonl" > "$RPT/eval_cases.jsonl"

python scripts/aggregate_round4.py
echo "FINALIZE DONE ($(date '+%H:%M:%S'))"
