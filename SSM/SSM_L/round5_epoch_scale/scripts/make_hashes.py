"""Round5 hashes: config + data (manifest/indices) sha256, and all encoder checkpoint sha256.

Writes round5_epoch_scale/reports/hashes.json + checkpoint_sha256.csv.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R5 = ROOT / 'round5_epoch_scale'
R3 = ROOT / 'round3_scale_transfer'
REP = R5 / 'reports'
BACKBONES = ('G', 'G_Qlearn', 'Q')
CORPORA = ('f1', 'f2')
SEEDS = (17, 42, 2026)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    REP.mkdir(parents=True, exist_ok=True)
    configs = {
        'pretrain.json': ROOT / 'configs' / 'pretrain.json',
        'manifest_f1.json': R3 / 'configs' / 'manifest_f1.json',
        'manifest_f2.json': R3 / 'configs' / 'manifest_f2.json',
        'f1_indices.npy': R3 / 'data' / 'f1_indices.npy',
        'f2_train_idx.npy': R3 / 'data' / 'f2_train_idx.npy',
        'f2_val_idx.npy': R3 / 'data' / 'f2_val_idx.npy',
        'f2_comm_flags.npy': R3 / 'data' / 'f2_comm_flags.npy',
        'downstream_nocfo_rml.json': ROOT / 'round4_no_cfo_qstudy' / 'configs' / 'downstream_nocfo_rml.json',
        'downstream_nocfo_radar.json': ROOT / 'round4_no_cfo_qstudy' / 'configs' / 'downstream_nocfo_radar.json',
        'rml_manifest.json': ROOT / 'configs' / 'data.json',
        'radar_manifest.json': ROOT / 'round2_overnight' / 'configs' / 'radchar.json',
    }
    config_hashes = {k: sha256(p) for k, p in configs.items() if p.exists()}

    # encoder checkpoints (new 50ep) + shared_init caches
    rows = []
    for corpus in CORPORA:
        for bb in BACKBONES:
            for s in SEEDS:
                run = R5 / 'runs' / 'pretrain' / corpus / f'{bb}_seed{s}'
                for ck in ('ckpt_step7800.pt', 'ckpt_epoch20.pt', 'ckpt_epoch30.pt',
                           'ckpt_epoch40.pt', 'ckpt_epoch50.pt', 'encoder_final.pt'):
                    p = run / ck
                    if p.exists():
                        rows.append({'corpus': corpus, 'backbone': bb, 'seed': s,
                                     'checkpoint': ck, 'sha256': sha256(p)})
            # shared_init (random init, reused from round3)
            p = R5 / 'runs' / 'pretrain' / corpus / 'shared_init' / f'{bb}_seed{s}.pt'
            # per-seed init differs only for s; but shared_init key is per (bb,seed)
            pass
    # shared_init hashes (one per bb/seed, identical across corpora)
    init_rows = []
    for bb in BACKBONES:
        for s in SEEDS:
            p = R5 / 'runs' / 'pretrain' / 'f1' / 'shared_init' / f'{bb}_seed{s}.pt'
            if p.exists():
                init_rows.append({'backbone': bb, 'seed': s, 'shared_init_sha256': sha256(p)})

    (REP / 'hashes.json').write_text(json.dumps(
        {'config_and_data_sha256': config_hashes,
         'shared_init_sha256': init_rows}, indent=2))
    import csv
    with open(REP / 'checkpoint_sha256.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['corpus', 'backbone', 'seed', 'checkpoint', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    print(f'[hashes] {len(config_hashes)} config/data, {len(rows)} checkpoints, '
          f'{len(init_rows)} shared-init')


if __name__ == '__main__':
    main()
