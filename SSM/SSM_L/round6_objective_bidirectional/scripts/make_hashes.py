"""Round6 hashes: config + data (manifest/indices) sha256, and all encoder checkpoint sha256.

Writes round6_objective_bidirectional/reports/hashes.json + checkpoint_sha256.csv.
Covers the 15 new round6 runs (G-MAE/Q-MAE reused from round5, not re-listed here except as
shared-init references).
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R6 = ROOT / 'round6_objective_bidirectional'
R5 = ROOT / 'round5_epoch_scale'
R3 = ROOT / 'round3_scale_transfer'
REP = R6 / 'reports'
SEEDS = (17, 42, 2026)
GROUPS = [('G', 'denoise'), ('G', 'joint'), ('Q', 'denoise'), ('Q', 'joint'), ('Q_bi', 'mae')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    REP.mkdir(parents=True, exist_ok=True)
    configs = {
        'pretrain.json': R6 / 'configs' / 'pretrain.json',
        'downstream_nocfo_rml.json': R6 / 'configs' / 'downstream_nocfo_rml.json',
        'downstream_nocfo_radar.json': R6 / 'configs' / 'downstream_nocfo_radar.json',
        'manifest_f1.json': R3 / 'configs' / 'manifest_f1.json',
        'f1_indices.npy': R3 / 'data' / 'f1_indices.npy',
        'rml_manifest.json': ROOT / 'configs' / 'data.json',
        'radar_manifest.json': ROOT / 'round2_overnight' / 'configs' / 'radchar.json',
    }
    config_hashes = {k: sha256(p) for k, p in configs.items() if p.exists()}

    rows = []
    for bb, task in GROUPS:
        for s in SEEDS:
            run = R6 / 'runs' / 'pretrain' / 'f1' / f'{bb}_{task}_seed{s}'
            for ck in ('ckpt_epoch10.pt', 'ckpt_epoch20.pt', 'ckpt_epoch30.pt',
                       'ckpt_epoch40.pt', 'ckpt_epoch50.pt', 'encoder_final.pt'):
                p = run / ck
                if p.exists():
                    rows.append({'backbone': bb, 'task': task, 'seed': s,
                                 'checkpoint': ck, 'sha256': sha256(p)})

    init_rows = []
    for bb, task in GROUPS:
        for s in SEEDS:
            # Q_bi shared-init is cached under R6 shared_init; G/Q reuse round5's.
            if bb == 'Q_bi':
                p = R6 / 'runs' / 'pretrain' / 'f1' / 'shared_init' / f'Q_bi_seed{s}.pt'
                if p.exists():
                    init_rows.append({'backbone': bb, 'seed': s,
                                      'shared_init_sha256': sha256(p)})
            else:
                p = R5 / 'runs' / 'pretrain' / 'f1' / 'shared_init' / f'{bb}_seed{s}.pt'
                if p.exists():
                    init_rows.append({'backbone': bb, 'seed': s,
                                      'shared_init_sha256': sha256(p),
                                      'note': 'reused from round5'})

    (REP / 'hashes.json').write_text(json.dumps(
        {'config_and_data_sha256': config_hashes,
         'shared_init_sha256': init_rows}, indent=2))
    with open(REP / 'checkpoint_sha256.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['backbone', 'task', 'seed', 'checkpoint', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    print(f'[hashes] {len(config_hashes)} config/data, {len(rows)} checkpoints, '
          f'{len(init_rows)} shared-init')


if __name__ == '__main__':
    main()
