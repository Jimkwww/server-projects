# Round-5 aux-branch diagnostics (learned recurrence parameters from best.pt)

q[m] = a·q[m-1] + b·s[m] over the full 1024-sample raw IQ; tau = 1/softplus(raw_alpha),
theta is the per-channel rotation, |b| the input coupling. R_corr: 8 groups of 2-dim
real state with per-step decay rho = sigmoid(logit) ∈ (0,1) (spectral norm < 1 enforced).

## Complex recurrence (Q_linear / Q_corr)
| variant | seed | tau (min/mean/max) | \|b\| (min/mean/max) | theta (deg) |
|---|---|---|---|---|
| Q_linear | 17 | 2.01 / 32.49 / 110.20 | 0.057 / 0.149 / 0.302 | [-180.0, -134.0, -91.0, -44.0, -0.0, 48.0, 92.0, 136.0] |
| Q_linear | 42 | 2.01 / 34.40 / 120.99 | 0.004 / 0.062 / 0.270 | [-178.0, -134.0, -93.0, -45.0, -0.0, 49.0, 89.0, 136.0] |
| Q_linear | 2026 | 2.04 / 33.91 / 120.25 | 0.011 / 0.103 / 0.219 | [-180.0, -134.0, -93.0, -44.0, -0.0, 47.0, 89.0, 136.0] |
| Q_corr | 17 | 1.91 / 31.82 / 114.95 | 0.009 / 0.149 / 0.215 | [-179.0, -134.0, -86.0, -42.0, 2.0, 43.0, 89.0, 133.0] |
| Q_corr | 42 | 1.87 / 31.38 / 112.46 | 0.001 / 0.077 / 0.317 | [-179.0, -134.0, -86.0, -41.0, -0.0, 42.0, 87.0, 134.0] |
| Q_corr | 2026 | 1.89 / 31.30 / 111.97 | 0.005 / 0.116 / 0.259 | [-179.0, -134.0, -86.0, -42.0, -1.0, 42.0, 87.0, 134.0] |

## Real recurrence (R_corr)
| variant | seed | rho (min/mean/max) | A_raw ‖·‖₂ (min/mean/max) | A_eff ‖·‖₂ (max) | B ‖·‖_F (min/mean/max) |
|---|---|---|---|---|---|
| R_corr | 17 | 0.5942 / 0.8747 / 0.9916 | 1.004 / 1.038 / 1.065 | 0.9916 | 0.106 / 0.218 / 0.376 |
| R_corr | 42 | 0.5897 / 0.8746 / 0.9916 | 0.920 / 1.005 / 1.077 | 0.9916 | 0.043 / 0.116 / 0.417 |
| R_corr | 2026 | 0.5916 / 0.8744 / 0.9916 | 0.920 / 1.023 / 1.080 | 0.9916 | 0.054 / 0.173 / 0.288 |
