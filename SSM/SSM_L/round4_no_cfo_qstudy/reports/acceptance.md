# Round-4 acceptance

## Acceptance checks

- [x] no-CFO in all round4 configs — bad=[]
- [x] every best.pt has DONE.json — missing=0
- [x] run_manifest.csv exists — /home/user_wzk/wzk/projects/SSM/SSM_L/round4_no_cfo_qstudy/reports/run_manifest.csv
- [x] no failed runs in manifest — failed=0
- [x] fixed-Q aux hash verified for A/B — bad=[]
- [x] all runs have val_predictions.npz — missing=0
- [x] radar eval-on-test once + eval_predictions — bad=0
- [x] finetune provenance records pretrained_sha256 — missing=0
- [x] radar 512pt/5cls, RML 1024pt/24cls — radar=512/5 rml=1024/24
- [x] MAE mask applied before aux descriptor — encoder2=True, gcorr=True

