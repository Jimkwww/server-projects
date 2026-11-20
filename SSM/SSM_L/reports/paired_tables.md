# SSM_L round-1 results

## probe — clean val OA (%) per seed, mean±std over seeds [17, 42, 2026]

| group | seed 17 | seed 42 | seed 2026 | mean±std | Macro-F1 (mean) | low-SNR OA (mean) | high-SNR OA (mean) |
|---|---|---|---|---|---|---|---|
| G_mae | 21.47 | 21.19 | 20.74 | 21.13±0.37 | 19.07 | 5.18 | 32.83 |
| G_denoise | 16.19 | 17.28 | 15.58 | 16.35±0.86 | 12.89 | 4.75 | 24.85 |
| Q_mae | 26.86 | 26.38 | 25.96 | 26.40±0.45 | 23.36 | 10.93 | 37.74 |
| Q_denoise | 22.60 | 20.51 | 21.83 | 21.65±1.05 | 17.65 | 7.83 | 31.78 |

### paired differences (percentage points, per seed then mean)

| difference | seed 17 | seed 42 | seed 2026 | mean |
|---|---|---|---|---|
| G-denoise − G-MAE | -5.29 | -3.91 | -5.16 | -4.79 |
| Q-denoise − Q-MAE | -4.26 | -5.87 | -4.13 | -4.75 |
| Q-denoise − G-denoise | +6.41 | +3.24 | +6.25 | +5.30 |
| (Q-denoise − Q-MAE) − (G-denoise − G-MAE) | +1.03 | -1.96 | +1.03 | +0.03 |

## finetune — clean val OA (%) per seed, mean±std over seeds [17, 42, 2026]

| group | seed 17 | seed 42 | seed 2026 | mean±std | Macro-F1 (mean) | low-SNR OA (mean) | high-SNR OA (mean) |
|---|---|---|---|---|---|---|---|
| G_mae | 51.60 | 47.34 | 46.09 | 48.34±2.89 | 47.66 | 15.13 | 72.70 |
| G_denoise | 46.35 | 52.21 | 46.12 | 48.23±3.45 | 47.43 | 15.05 | 72.56 |
| Q_mae | 49.90 | 46.83 | 49.13 | 48.62±1.60 | 47.51 | 17.20 | 71.67 |
| Q_denoise | 44.10 | 46.83 | 44.17 | 45.03±1.55 | 43.23 | 16.89 | 65.67 |

### paired differences (percentage points, per seed then mean)

| difference | seed 17 | seed 42 | seed 2026 | mean |
|---|---|---|---|---|
| G-denoise − G-MAE | -5.26 | +4.87 | +0.03 | -0.12 |
| Q-denoise − Q-MAE | -5.80 | +0.00 | -4.97 | -3.59 |
| Q-denoise − G-denoise | -2.24 | -5.38 | -1.96 | -3.19 |
| (Q-denoise − Q-MAE) − (G-denoise − G-MAE) | -0.54 | -4.87 | -5.00 | -3.47 |

## supervised baselines (reused from old project) — clean val OA (%)

| baseline | seed 17 | seed 42 | seed 2026 | mean±std |
|---|---|---|---|---|
| G_scratch | 44.04 | 44.52 | 43.97 | 44.18±0.30 |
| Q_scratch | 45.00 | 46.12 | 45.19 | 45.44±0.60 |
