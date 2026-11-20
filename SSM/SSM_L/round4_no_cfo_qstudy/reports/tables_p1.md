# Round-4 P1 tables

## P1: Q learning-phase 2×2 (G_Qlearn, F0-MAE) val OA

### RML

| setting | seed17 | seed42 | seed2026 | mean ± sample-std |
|---|---|---|---|---|
| A_fixedq_fixedq | 0.3978 | 0.4087 | 0.4186 | 0.4083 ± 0.0104 |
| B_learnq_fixedq | 0.3968 | 0.3984 | 0.4051 | 0.4001 ± 0.0044 |
| C_fixedq_learnq | 0.3990 | 0.4038 | 0.4218 | 0.4082 ± 0.0120 |
| D_learnq_learnq | 0.3910 | 0.4038 | 0.3897 | 0.3949 ± 0.0078 |

Paired deltas vs D (learn-Q pretrain + learn-Q finetune):

- A_fixedq_fixedq: per-seed ['0.0067', '0.0048', '0.0288'], mean +0.0135 ± 0.0134
- B_learnq_fixedq: per-seed ['0.0058', '-0.0054', '0.0154'], mean +0.0052 ± 0.0104
- C_fixedq_learnq: per-seed ['0.0080', '0.0000', '0.0321'], mean +0.0134 ± 0.0167

References:

- G_scratch: 0.3869 ± 0.0056
- G_corr_scratch: 0.3732 ± 0.0050
- G_corr_mae: 0.3785 ± 0.0114
- G_corr_probe: 0.1793 ± 0.0098

### RADAR

| setting | seed17 | seed42 | seed2026 | mean ± sample-std |
|---|---|---|---|---|
| A_fixedq_fixedq | 0.8692 | 0.8634 | 0.8652 | 0.8659 ± 0.0030 |
| B_learnq_fixedq | 0.8678 | 0.8617 | 0.8672 | 0.8656 ± 0.0034 |
| C_fixedq_learnq | 0.8643 | 0.8649 | 0.8663 | 0.8652 ± 0.0010 |
| D_learnq_learnq | 0.8641 | 0.8642 | 0.8668 | 0.8650 ± 0.0015 |

Paired deltas vs D (learn-Q pretrain + learn-Q finetune):

- A_fixedq_fixedq: per-seed ['0.0051', '-0.0008', '-0.0016'], mean +0.0009 ± 0.0037
- B_learnq_fixedq: per-seed ['0.0037', '-0.0025', '0.0004'], mean +0.0005 ± 0.0031
- C_fixedq_learnq: per-seed ['0.0002', '0.0007', '-0.0005'], mean +0.0001 ± 0.0006

References:

- G_scratch: 0.8547 ± 0.0020
- G_corr_scratch: 0.8522 ± 0.0011
- G_corr_mae: 0.8624 ± 0.0037
- G_corr_probe: 0.6617 ± 0.0062

