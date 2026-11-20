# Round-5 efficiency benchmark (eval mode, CUDA warmup+sync; L=1024)

Real-scalar params (complex ×2). Eager PyTorch O(N log N) main-path scan, not fused;
aux branch runs on the continuous 1024-sample IQ (ComplexRecurBranch via parallel_scan;
RealRecurBranch via an explicit 1024-step loop).

| variant | params | batch1 latency (ms) | batch64 throughput (sps) | infer peak mem (MiB) |
|---|---|---|---|---|
| S1 | 2216936 | 2.903 | 9031.3 | 1429.6 |
| S1_corr | 2217464 | 2.749 | 8282.0 | 1427.1 |
| G | 2251592 | 1.004 | 15088.4 | 654.6 |
| G_corr | 2252120 | 1.209 | 16270.0 | 654.6 |
| R0 | 2216760 | 2.737 | 8557.5 | 1428.1 |
| C0 | 2215352 | 2.731 | 8455.3 | 1427.1 |
| Rb | 2216936 | 2.716 | 8522.0 | 1427.1 |
| Cb | 2215528 | 2.740 | 8458.3 | 1427.1 |
| Q_linear | 2219784 | 3.169 | 7976.5 | 1504.0 |
| Q_corr | 2219784 | 3.283 | 7868.1 | 1513.5 |
| R_corr | 2219824 | 43.964 | 1264.4 | 1445.1 |
