# Revision-1 Section 3: unified frozen readout

Protocol: frozen encoder_final.pt -> mean-pool -> StandardScaler(fit train) + L2 LogisticRegression(C=1, lbfgs, max_iter=2000). No eval-based hyperparameter selection. RML main = val (test sealed); radar main = eval (dev used only for the side table).


## RML — main split = val

| encoder | seed17 | seed42 | seed2026 | mean ± sample-std | macro-F1 mean |

|---|---|---|---|---|---|

| G_Qlearn_fixedq_mae | 0.3029 | 0.2894 | 0.2897 | 0.2940 ± 0.0077 | 0.2757 |

| G_Qlearn_learnq_mae | 0.2846 | 0.2804 | 0.2878 | 0.2843 ± 0.0037 | 0.2646 |

| G_corr_mae | 0.2660 | 0.2740 | 0.2856 | 0.2752 ± 0.0098 | 0.2543 |

| G_mae | 0.3208 | 0.3202 | 0.3106 | 0.3172 ± 0.0057 | 0.3046 |


Paired per-seed deltas (fixed-Q − learn-Q):

- G_Qlearn fixed-Q − learn-Q: ['0.0183', '0.0090', '0.0019'], mean +0.0097 ± 0.0082


Paired per-seed deltas (G_corr − G):

- G_corr − G: ['-0.0548', '-0.0462', '-0.0250'], mean -0.0420 ± 0.0153


## RADAR — main split = test

| encoder | seed17 | seed42 | seed2026 | mean ± sample-std | macro-F1 mean |

|---|---|---|---|---|---|

| G_Qlearn_fixedq_mae | 0.8046 | 0.8013 | 0.7960 | 0.8006 ± 0.0043 | 0.8003 |

| G_Qlearn_learnq_mae | 0.8070 | 0.8054 | 0.8002 | 0.8042 ± 0.0036 | 0.8038 |

| G_corr_mae | 0.7996 | 0.7900 | 0.7969 | 0.7955 ± 0.0050 | 0.7954 |

| G_mae | 0.7675 | 0.7575 | 0.7673 | 0.7641 ± 0.0057 | 0.7640 |


Side table (dev, model selection split, NOT main):

| encoder | seed17 | seed42 | seed2026 | mean ± sample-std |

|---|---|---|---|---|

| G_Qlearn_fixedq_mae | 0.8039 | 0.7944 | 0.7916 | 0.7966 ± 0.0064 |

| G_Qlearn_learnq_mae | 0.8080 | 0.8016 | 0.7944 | 0.8013 ± 0.0068 |

| G_corr_mae | 0.7918 | 0.7861 | 0.7947 | 0.7909 ± 0.0044 |

| G_mae | 0.7653 | 0.7469 | 0.7720 | 0.7614 ± 0.0130 |


Paired per-seed deltas (fixed-Q − learn-Q):

- G_Qlearn fixed-Q − learn-Q: ['-0.0024', '-0.0041', '-0.0042'], mean -0.0036 ± 0.0010


Paired per-seed deltas (G_corr − G):

- G_corr − G: ['0.0321', '0.0325', '0.0296'], mean +0.0314 ± 0.0016
