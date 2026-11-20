#!/bin/bash
set -uo pipefail
BASE="https://hf-mirror.com"
W=/data1/user_wzk/LMO/weights
Q=/data1/user_wzk/LMO/models/Qwen
D=/data1/user_wzk/LMO/data/interference
mkdir -p "$W" "$Q" "$D"

dl() { curl -L --retry 8 --retry-delay 3 -C - -sS -o "$2" "$1" && echo "OK  $(basename $2)"; }

echo "=== [1/3] LEMMO_Recognition.pt (683MB) ==="
dl "$BASE/datasets/DaftJun/RFLAS/resolve/main/ckpt/LEMMO_Recognition.pt" "$W/LEMMO_Recognition.pt"

echo "=== [2/3] Qwen3.5-4B (~9.3GB) ==="
R="$BASE/Qwen/Qwen3.5-4B/resolve/main"
for f in config.json tokenizer_config.json tokenizer.json vocab.json merges.txt model.safetensors.index.json model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors preprocessor_config.json chat_template.jinja; do
  dl "$R/$f" "$Q/$f"
done

echo "=== [3/3] small interference parquet (schema inspect) ==="
dl "$BASE/datasets/DaftJun/RFLAS/resolve/main/data/interference/gnss_standardized/gnss_jamming_test.parquet" "$D/gnss_jamming_test.parquet"
dl "$BASE/datasets/DaftJun/RFLAS/resolve/main/data/interference/owl_standardized/owl_init_test.parquet" "$D/owl_init_test.parquet"

echo "=== ALL DOWNLOADS DONE ==="
du -sh "$W" "$Q" "$D"
