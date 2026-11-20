# Revision-1 Section 4: SSM memory diagnosis (corrected)

A group = pretrained (no downstream supervision) encoders, two frozen readouts (fixed on full-state train vs per-K refit). B group = finetuned models, original head, per-K OA vs full. Reset = zero the recurrence state every K tokens (all layers share the boundary; zero-init per segment, NOT detach). `full` = no reset.


## A group (pretrained encoder + LR readout)

Main contrast: **fixed** readout (frozen scaler/LR fit on full-state train) vs **refit** readout (per-K scaler/LR refit). Val split (RML val / radar dev).


### RML

| init | readout | full | K1 | K4 | K8 | K16 | K32 | K64 |

|---|---|---|---|---|---|---|---|---|

| random | fixed | 0.223 | 0.042 | 0.042 | 0.042 | 0.060 | 0.082 | 0.145 |

| random | refit | 0.223 | 0.224 | 0.243 | 0.235 | 0.227 | 0.223 | 0.221 |

| mae | fixed | 0.328 | 0.058 | 0.066 | 0.087 | 0.119 | 0.190 | 0.276 |

| mae | refit | 0.328 | 0.315 | 0.314 | 0.317 | 0.320 | 0.324 | 0.325 |

| denoise | fixed | 0.287 | 0.052 | 0.051 | 0.056 | 0.074 | 0.121 | 0.207 |

| denoise | refit | 0.287 | 0.253 | 0.271 | 0.276 | 0.282 | 0.281 | 0.285 |


### RADAR

| init | readout | full | K1 | K4 | K8 | K16 | K32 | K64 |

|---|---|---|---|---|---|---|---|---|

| random | fixed | 0.729 | 0.230 | 0.200 | 0.200 | 0.218 | 0.262 | 0.729 |

| random | refit | 0.729 | 0.781 | 0.764 | 0.754 | 0.728 | 0.716 | 0.729 |

| mae | fixed | 0.785 | 0.201 | 0.201 | 0.203 | 0.216 | 0.308 | 0.785 |

| mae | refit | 0.785 | 0.758 | 0.748 | 0.758 | 0.759 | 0.769 | 0.785 |

| denoise | fixed | 0.818 | 0.245 | 0.251 | 0.265 | 0.314 | 0.436 | 0.818 |

| denoise | refit | 0.818 | 0.800 | 0.800 | 0.801 | 0.801 | 0.807 | 0.818 |


## B group (finetuned model, original head)

`delta` = per-K OA − full OA (negative = worse than full). Mean over 3 seeds.


### RML (full OA → K1/K4/K8/K16/K32/K64)

| init | full | K1 | K4 | K8 | K16 | K32 | K64 |

|---|---|---|---|---|---|---|---|

| scratch | 0.381 | 0.060 | 0.084 | 0.128 | 0.178 | 0.291 | 0.349 |

| mae | 0.410 | 0.052 | 0.065 | 0.099 | 0.174 | 0.304 | 0.388 |

| denoise | 0.388 | 0.061 | 0.086 | 0.100 | 0.144 | 0.237 | 0.348 |


### RADAR (full OA → K1/K4/K8/K16/K32/K64)

| init | full | K1 | K4 | K8 | K16 | K32 | K64 |

|---|---|---|---|---|---|---|---|

| scratch | 0.871 | 0.210 | 0.246 | 0.346 | 0.546 | 0.790 | 0.871 |

| mae | 0.871 | 0.294 | 0.390 | 0.570 | 0.704 | 0.843 | 0.871 |

| denoise | 0.875 | 0.296 | 0.473 | 0.647 | 0.739 | 0.850 | 0.875 |


## Consistency checks (per encoder, all passed)

`K_none_matches_original_maxdiff = 0.0`; `segment_vs_serial_* ≤ ~1.2e-6`; `K_eq_N_equals_full = 0.0`; `K1_equals_b = 0.0`. Full values in `memory_A_raw.json` / `memory_B_raw.json`.
