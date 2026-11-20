# Round-4 launch / resume / rebuild commands

All paths relative to the SSM_L root `/home/user_wzk/wzk/projects/SSM/SSM_L`.
Everything runs inside the `SFM` conda env (`torch 2.14.0+cu126`, 3×A100 40GB).

## 1. Launch the full job queue (P0 + P1)

```bash
cd /home/user_wzk/wzk/projects/SSM/SSM_L
setsid tmux new-session -d -s r4_orch -c "$PWD" \
  "source /home/user_wzk/miniconda3/etc/profile.d/conda.sh && conda activate SFM && \
   python round4_no_cfo_qstudy/scripts/orchestrate.py > round4_no_cfo_qstudy/logs/orchestrate.out 2>&1"
```

- The orchestrator builds 105 jobs, runs them on GPUs {0,1,2} with a FIFO queue, skips any
  job whose `DONE.json` exists (resumable), and writes `reports/run_manifest.csv`.
- Watch progress: `tail -f round4_no_cfo_qstudy/logs/orchestrate.out`.

## 2. Resume after interruption / crash

The queue is `DONE.json`-based, so just re-run the same command. Already-completed jobs are
skipped; a job marked `failed` in `run_manifest.csv` is NOT auto-retried — inspect its log
(`round4_no_cfo_qstudy/logs/<id>.log`), fix the cause, delete that row's `failed` status in
`run_manifest.csv` (or delete its `DONE.json` if it was a stale partial), then relaunch.

```bash
# example: retry one failed job manually
source /home/user_wzk/miniconda3/etc/profile.d/conda.sh && conda activate SFM
python -m round4_no_cfo_qstudy.downstream_r4 --mode finetune \
  --config round4_no_cfo_qstudy/configs/downstream_nocfo_rml.json \
  --manifest configs/data.json --backbone G \
  --pretrained runs/pretrain/G_mae_seed17/encoder_final.pt \
  --out round4_no_cfo_qstudy/runs/rml/mae/G_seed17 --seed 17 --device cuda:0
```

## 3. Rebuild (from a clean checkout)

1. Restore `round4_no_cfo_qstudy/` (code + configs + reports + logs) and the three read-only
   reference trees it imports: `ssm_exp/`, `pretrain/`, `round2_overnight/`, plus the two data
   manifests (`configs/data.json`, `round2_overnight/configs/radchar.json`) and the F0 pretrain
   weights (see `reports/reuse_inventory.csv` for the exact checkpoint paths).
2. Point the configs' data paths at the local datasets (RML2018A h5, RadChar-Tiny h5).
3. Re-run the orchestrator (section 1). `DONE.json` markers are intentionally NOT committed,
   so the queue will re-run everything — or restore `runs/**/best.pt` + `DONE.json` to skip.

## 4. Ancillary scripts

```bash
# P0 appendix: radar simple baselines (flattened IQ + fixed physical features, sklearn LR)
python round4_no_cfo_qstudy/scripts/radar_baselines.py \
  --manifest round2_overnight/configs/radchar.json \
  --out round4_no_cfo_qstudy/runs/radar_baselines

# P2: SSM effective-memory diagnosis (run after S1plain downstream runs exist)
python round4_no_cfo_qstudy/scripts/p2_ssm_memory.py --task rml --device cuda:0
python round4_no_cfo_qstudy/scripts/p2_ssm_memory.py --task radar --device cuda:1

# Report aggregation (tables + acceptance + results_summary.json)
python round4_no_cfo_qstudy/scripts/make_report.py
```

## 5. Packaging (weights stay on server)

```bash
python round4_no_cfo_qstudy/scripts/package.py
# -> round4_no_cfo_qstudy/round4_package.tar.gz  (code + configs + reports + predictions, no weights)
```
