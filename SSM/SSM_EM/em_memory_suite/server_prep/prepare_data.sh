#!/usr/bin/env bash
# 数据准备：RML2018A + RadChar → emmem npy manifest 格式。
# 复用 SSM_L 项目已验证的 cls_dev / f1 / radchar 划分，不重做划分。
set -e
cd "$(dirname "$0")/.."
PY="conda run -n SFM python"

# ---- 1. groups 数组（样本级唯一；RML2018A/RadChar 均为独立模拟样本，无同源 capture group）----
$PY - <<'PY'
import numpy as np
np.save('data/rml2018_groups.npy', np.arange(2555904, dtype=np.int64))
np.save('data/radchar_groups.npy', np.arange(50000, dtype=np.int64))
print('[prep] groups arrays written')
PY

# ---- 2. splits JSON（引用 SSM_L 已有索引）----
SL=/home/user_wzk/wzk/projects/SSM/SSM_L
$PY - <<PY
import json
d = {
 'amc': {
   'train': '$SL/data/cls_dev_train.npy',
   'val': '$SL/data/cls_dev_val.npy',
   'test': '$SL/data/cls_dev_test.npy'},
 'f1': {
   'train': '$SL/round3_scale_transfer/data/f1_indices.npy',
   'val': '$SL/data/cls_dev_val.npy'},
 'radchar': {
   'train': '$SL/round2_overnight/data/radchar_train.npy',
   'val': '$SL/round2_overnight/data/radchar_dev.npy',
   'test': '$SL/round2_overnight/data/radchar_eval.npy'},
}
for k,v in d.items():
    json.dump(v, open(f'server_prep/splits_{k}.json','w'))
print('[prep] splits json written')
PY

# ---- 3. convert ----
RML=/data1/user_wzk/datasets/RML2018A/GOLD_XYZ_OSC.0001_1024.hdf5
RAD=/data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5
$PY -m tools.convert_rml_h5 --h5 $RML --splits server_prep/splits_amc.json \
  --groups data/rml2018_groups.npy --out data/rml2018_clsdev \
  --dataset-id rml2018a_clsdev \
  --provenance 'RML2018A (Deepsig 2018.01a), 24-class AMC. Splits reused from SSM_L cls_dev (train 19968/val 3120/test 8112), (mod,SNR) stratified. Samples independently simulated; no shared source-capture grouping, groups=per-sample unique id. normalization=none (raw IQ, amplitude retained).'
$PY -m tools.convert_rml_h5 --h5 $RML --splits server_prep/splits_f1.json \
  --groups data/rml2018_groups.npy --out data/rml2018_f1 \
  --dataset-id rml2018a_f1 \
  --provenance 'RML2018A F1 pretrain corpus (100k, (mod,SNR) stratified), reused from SSM_L round3 f1_indices. No overlap with cls_dev val/test (verified). Samples independently simulated; groups=per-sample unique id. normalization=none.'
$PY -m tools.convert_rml_h5 --h5 $RAD --splits server_prep/splits_radchar.json \
  --groups data/radchar_groups.npy --out data/radchar_tiny \
  --dataset-id radchar_tiny \
  --provenance 'RadChar-Tiny RML radar waveforms (5-class), reused from SSM_L round2 radchar split (train 30000/dev 10000/eval 10000). Samples independently simulated; groups=per-sample unique id. normalization=none.'
echo '[prep] ALL DATA CONVERTED'
