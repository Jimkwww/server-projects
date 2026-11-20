# Revision-1 Section 6: Q 2D fairness audit (no rerun)

## 1. fixed-Q vs learn-Q pretrain initial weights
| seed | encoder | filter | buffers | aux stats | decoder init |
|---|---|---|---|---|---|
| 17 | True | True | True | True | True |
| 42 | True | True | True | True | True |
| 2026 | True | True | True | True | True |

Both pretrain arms start from byte-identical encoder + decoder + aux stats; the only difference is `freeze_aux` (requires_grad on the filter), which does not change values.

## 2. pretrain config round2 (learn-Q) vs round4 (fixed-Q)
- model config equal: True
- train config diffs: none (identical)

## 3. downstream A/B/C/D freezing
| task:setting | expected | aux_frozen_verified | hash before==after |
|---|---|---|---|
| rml:A_fixedq_fixedq:seed17 | frozen | True | True |
| rml:A_fixedq_fixedq:seed42 | frozen | True | True |
| rml:A_fixedq_fixedq:seed2026 | frozen | True | True |
| rml:B_learnq_fixedq:seed17 | frozen | True | True |
| rml:B_learnq_fixedq:seed42 | frozen | True | True |
| rml:B_learnq_fixedq:seed2026 | frozen | True | True |
| rml:C_fixedq_learnq:seed17 | learn | None | None |
| rml:C_fixedq_learnq:seed42 | learn | None | None |
| rml:C_fixedq_learnq:seed2026 | learn | None | None |
| rml:D_learnq_learnq:seed17 | learn | None | None |
| rml:D_learnq_learnq:seed42 | learn | None | None |
| rml:D_learnq_learnq:seed2026 | learn | None | None |
| radar:A_fixedq_fixedq:seed17 | frozen | True | True |
| radar:A_fixedq_fixedq:seed42 | frozen | True | True |
| radar:A_fixedq_fixedq:seed2026 | frozen | True | True |
| radar:B_learnq_fixedq:seed17 | frozen | True | True |
| radar:B_learnq_fixedq:seed42 | frozen | True | True |
| radar:B_learnq_fixedq:seed2026 | frozen | True | True |
| radar:C_fixedq_learnq:seed17 | learn | None | None |
| radar:C_fixedq_learnq:seed42 | learn | None | None |
| radar:C_fixedq_learnq:seed2026 | learn | None | None |
| radar:D_learnq_learnq:seed17 | learn | None | None |
| radar:D_learnq_learnq:seed42 | learn | None | None |
| radar:D_learnq_learnq:seed2026 | learn | None | None |

## 4. C/D filter actually updates
- rml: C filter max|Δ| vs fixed-Q pretrain = 0.1264; D filter max|Δ| vs learn-Q pretrain = 0.1369
- radar: C filter max|Δ| vs fixed-Q pretrain = 0.0912; D filter max|Δ| vs learn-Q pretrain = 0.1806

## 5. resume behavior
- resume restores aux/phys from the checkpoint (ck.get("aux"/"phys")), and model.load_state_dict(ck["model"]) restores aux_mean/aux_std buffers; NO stat re-fit on resume. A fresh fit can occur only if the shared_init cache is missing, and is then overwritten by the checkpoint load.

## 6. unverifiable / not re-run
- round2 learn-Q pretrain used a different entry script (pretrain/engine.py) than round4 fixed-Q (pretrain_r4.py); optimizer param selection differs textually (round2: model.parameters(); round4: requires_grad filter) but is equivalent when no params are frozen, and the mask stream (seed+200000) and data order (seed) are identical. Full bit-level equivalence of the two training loops was not re-run.
- The exact decoder init RNG state was reconstructed and matched; a full forward/backward bit-match of the two engines was not re-executed (未核验).
