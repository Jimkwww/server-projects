#!/usr/bin/env bash
# 结果快照：把当前所有中间结果打包到 /data1（另一块盘），防 /home 盘损坏丢失。
# 只打包轻量结果（metrics/status/history/历史基线 json + 报告），不打包大权重（权重已在 runs/ 落盘且可恢复）。
set -e
cd "$(dirname "$0")/.."
STAMP=$(date '+%Y%m%d_%H%M%S')
DEST=/data1/user_wzk/em_memory_suite_backup
mkdir -p "$DEST"

SNAP_DIR="$DEST/snapshot_$STAMP"
mkdir -p "$SNAP_DIR"

# 1. 主实验的 eval metrics + status + history（轻量，结果核心）
for gpu in main_gpu1 main_gpu2; do
  if [ -d "results/$gpu/plan" ]; then
    mkdir -p "$SNAP_DIR/$gpu"
    cp results/$gpu/plan/status.json "$SNAP_DIR/$gpu/" 2>/dev/null || true
    # 打包 eval metrics（.json 很小）
    find results/$gpu/plan/eval -name 'metrics.json' 2>/dev/null | tar -czf "$SNAP_DIR/$gpu/eval_metrics.tar.gz" -T - 2>/dev/null || true
    # 打包 train history（jsonl 很小）
    find results/$gpu/plan/runs -name 'history.jsonl' 2>/dev/null | tar -czf "$SNAP_DIR/$gpu/history.tar.gz" -T - 2>/dev/null || true
    # 诊断 + benchmark
    cp -r results/$gpu/plan/diagnostics "$SNAP_DIR/$gpu/" 2>/dev/null || true
    cp -r results/$gpu/plan/bench "$SNAP_DIR/$gpu/" 2>/dev/null || true
  fi
done

# 2. 历史基线 json
mkdir -p "$SNAP_DIR/historical"
cp results/historical/*.json "$SNAP_DIR/historical/" 2>/dev/null || true

# 3. 元数据 + 变更记录
cp -r metadata "$SNAP_DIR/" 2>/dev/null || true
cp CHANGES.md "$SNAP_DIR/" 2>/dev/null || true
cp -r audit "$SNAP_DIR/" 2>/dev/null || true

# 4. 汇总计数
{
  echo "快照时间: $STAMP"
  echo "GPU1 eval: $(ls results/main_gpu1/plan/eval/*/metrics.json 2>/dev/null | wc -l) 个"
  echo "GPU2 eval: $(ls results/main_gpu2/plan/eval/*/metrics.json 2>/dev/null | wc -l) 个"
  echo "历史基线: $(ls results/historical/*.json 2>/dev/null | wc -l)/24"
} > "$SNAP_DIR/SUMMARY.txt"

echo "[snapshot] 保存到 $SNAP_DIR"
cat "$SNAP_DIR/SUMMARY.txt"
# 保留最近 10 个快照，清理更旧的
ls -1d "$DEST"/snapshot_* 2>/dev/null | sort | head -n -10 | xargs -r rm -rf
