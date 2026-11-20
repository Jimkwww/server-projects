# Round-4 Protocol (no-CFO downstream + Q learning-phase study)

## 1. Scope

- **P0** — no-CFO downstream evaluation on RML2018A (24-class) and RadChar-Tiny (5-class).
  Matrix: 4 structures (G / G_Qlearn / S1plain / Q) × 3 init (scratch / F0-MAE / F0-denoise)
  × 2 tasks × seeds {17, 42, 2026} = 72 slots. Reused where history is no-CFO; fresh otherwise.
- **P1** — Q learning-phase 2×2 control on Global+Q (G_Qlearn), F0-MAE only:
  A = fixed-Q pretrain + fixed-Q finetune; B = learn-Q pretrain + fixed-Q finetune;
  C = fixed-Q pretrain + learn-Q finetune; D = learn-Q pretrain + learn-Q finetune (= P0).
  G and G_corr serve as references (G_corr MAE pretrain, frozen readout, no-CFO scratch/MAE
  finetune are filled in here).
- **P2** — SSM effective-memory diagnosis on S1plain (state stats, segment reset K=1/4/16/64).

## 2. Hard constraints (from the task book)

1. Do NOT overwrite historical results, modify historical reports, or disturb running tasks.
2. **No CFO augmentation in any new training this round.** Raw-data noise/distortion is kept.
   Do NOT add phase-rotation / time-shift / any new augmentation. MAE masking and the denoise
   task's artificial noise still follow the original pretrain protocol.
3. Model definitions (code is authoritative):
   - G = plain IQ patch projection + Global Transformer.
   - G_Qlearn = plain projection + Q filtered-correlation branch + Global.
   - S1plain = plain projection + S1 trunk.
   - Q = plain projection + Q filtered-correlation branch + S1.
   - G_corr = plain projection + raw-IQ local complex correlation + Global.
4. "Fixed Q" means ONLY the filter parameters (decay/rotation/complex-input coefficients) are
   frozen; the fusion layer, the trunk, and (downstream) the head still train. It is NOT a
   frozen fusion layer and NOT a frozen encoder. Verified by recording the aux-branch parameter
   SHA256 before/after (`aux_frozen_verified`).
5. MAE masking is applied to the raw input BEFORE any auxiliary descriptor (Q corr or G_corr
   phys) is computed, so the descriptor cannot leak the target.

## 3. Data

- RML2018A: `configs/data.json` (h5, X [N,1024,2] NLC, Y one-hot 24, Z SNR).
  Splits train 19968 / val 3120 / test 8112 (SEALED; test is never used for selection).
- RadChar-Tiny: `round2_overnight/configs/radchar.json` (h5, 512-length IQ, 5 classes).
  Splits train 30000 / val(=dev) 10000 / test(=eval) 10000. Record-level random split;
  selection on val, eval on test once at the end.
- Both manifests `normalization=none`; `max_abs_norm` applied in-loop (idempotent).

## 4. Pretraining (round-4 new)

- MAE only. 50 epochs × ceil(19968/128)=156 steps = 7800 steps. batch 128, AdamW(lr 3e-4,
  wd 0.01, betas (0.9,0.999), eps 1e-8), cosine + 5% warmup, grad_clip 1.0, fp32.
- mask_ratio 0.5 (exactly round(N·0.5) masked patches, ≥1 visible), mask before front/aux.
- `pretrain_r4.py`: backbones via `build_encoder_r4`; `--freeze-aux` freezes only the Q filter.
- Output `encoder_final.pt` = epoch-50 encoder weights + aux/phys stats + provenance.

## 5. Downstream (round-4 new)

- 50 epochs, AdamW(lr 3e-4, wd 0.01), cosine + 5% warmup, grad_clip 1.0, fp32, batch 128.
- `augment_cfo_max == 0.0` is ENFORCED by `load_config` (raises otherwise). No CFO anywhere.
- Checkpoint selection: clean-val OA, earliest tie (`best.pt`). Radar selects on val(=dev);
  eval on test only once at the end (`--eval-split test`).
- scratch G_corr fits phys stats; scratch Q/G_Qlearn fits aux stats (rng-0 8192-sample subset).
- finetune/probe load the pretrain encoder (aux/phys buffers come from the checkpoint, never
  refit). `--freeze-aux` freezes only the Q filter and records the before/after hash.
- probe (frozen readout): encoder fully frozen, only a fresh linear head trains on precomputed
  mean-pooled features; encoder frozen hash recorded.

## 6. Reuse audit (why these are/aren't reusable)

- Historical SSM_L round1/2/3 finetunes + scratch ALL used CFO augmentation (0.002), which
  raised G clean val OA 0.3869 → 0.4418 (+5.5 pp). Therefore NOT reusable for no-CFO.
- Old SSM project `runs/clean` (G/S0/S1/S2/S2_concat) and `runs/round3` (G_corr) are
  `augment_cfo_max=0.0` → REUSABLE for RML scratch (G 0.3933/0.3837/0.3837; S1 0.3901/0.3760/
  0.3779; G_corr 0.3788/0.3712/0.3696).
- F0 pretrain weights (no-CFO, mask/denoise only) are REUSABLE: `runs/pretrain/{G,Q}_{mae,
  denoise}_seed*` and `round2_overnight/runs/pretrain/{G_Qlearn,S1plain}_{mae,denoise}_seed*`.

## 7. Deliverable notes

- Val is the main table; test (RML) stays sealed. Radar test is reported once.
- Per-sample predictions saved as `val_predictions.npz` / `eval_predictions.npz`
  (y, pred, snr, ids). Checkpoint SHA256 recorded in provenance (`pretrained_sha256`).
- Weights stay on the server; only code/config/reports/predictions are packaged.
