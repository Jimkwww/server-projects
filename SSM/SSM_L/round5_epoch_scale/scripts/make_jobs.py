"""Generate the round5 downstream job lists (readout + finetune).

Writes round5_epoch_scale/scripts/readout_jobs.txt (180) and finetune_jobs.txt (72).
Each line is a shell command with @@GPU@@ replaced by dispatch.py; ' done: <marker>' makes the
line idempotent (dispatch skips it if the marker exists).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R5 = ROOT / 'round5_epoch_scale'
SCRIPTS = R5 / 'scripts'
BACKBONES = ('G', 'G_Qlearn', 'Q')
CORPORA = ('f1', 'f2')
SEEDS = (17, 42, 2026)

CKPTS = [
    ('step7800', 'ckpt_step7800.pt'),
    ('epoch20', 'ckpt_epoch20.pt'),
    ('epoch30', 'ckpt_epoch30.pt'),
    ('epoch40', 'ckpt_epoch40.pt'),
    ('epoch50', 'encoder_final.pt'),
]

TASK_MANIFEST = {
    'rml': ROOT / 'configs' / 'data.json',
    'radar': ROOT / 'round2_overnight' / 'configs' / 'radchar.json',
}
TASK_CONFIG = {
    'rml': ROOT / 'round4_no_cfo_qstudy' / 'configs' / 'downstream_nocfo_rml.json',
    'radar': ROOT / 'round4_no_cfo_qstudy' / 'configs' / 'downstream_nocfo_radar.json',
}


def readout_jobs():
    lines = ['# round5 frozen readout jobs (90 checkpoints x 2 tasks = 180)']
    for corpus in CORPORA:
        for bb in BACKBONES:
            for seed in SEEDS:
                run_dir = R5 / 'runs' / 'pretrain' / corpus / f'{bb}_seed{seed}'
                for ckpt_name, ckpt_fn in CKPTS:
                    ckpt = run_dir / ckpt_fn
                    for task in ('rml', 'radar'):
                        out_json = (R5 / 'runs' / 'downstream' / 'readout' / corpus /
                                    f'{bb}_seed{seed}__{ckpt_name}__{task}.json')
                        cmd = (
                            f'python {SCRIPTS / "readout_r5.py"} '
                            f'--checkpoint {ckpt} --backbone {bb} --corpus {corpus} --seed {seed} '
                            f'--ckpt-name {ckpt_name} --task {task} --device cuda:@@GPU@@ '
                            f'--cache-dir {R5 / "runs" / "downstream" / "features"} '
                            f'--pred-dir {R5 / "runs" / "downstream" / "predictions"} '
                            f'--out-json {out_json} done: {out_json}'
                        )
                        lines.append(cmd)
    return lines


def finetune_jobs():
    lines = ['# round5 finetune jobs (18x2=36 for 50ep + 18x2=36 for 7800-step no-CFO = 72)']
    for which, pretrain_root in (('50ep', R5 / 'runs' / 'pretrain'),
                                 ('7800', ROOT / 'round3_scale_transfer' / 'runs' / 'pretrain_scale')):
        for corpus in CORPORA:
            for bb in BACKBONES:
                for seed in SEEDS:
                    pretrained = pretrain_root / corpus / f'{bb}_seed{seed}' / 'encoder_final.pt'
                    for task in ('rml', 'radar'):
                        out_dir = (R5 / 'runs' / 'downstream' / f'finetune_{which}' / task /
                                   f'{corpus}_{bb}_seed{seed}')
                        cmd = (
                            f'python -m round4_no_cfo_qstudy.downstream_r4 '
                            f'--mode finetune --config {TASK_CONFIG[task]} '
                            f'--manifest {TASK_MANIFEST[task]} --backbone {bb} '
                            f'--pretrained {pretrained} --out {out_dir} --seed {seed} '
                            f'--device cuda:@@GPU@@'
                        )
                        if task == 'radar':
                            cmd += ' --eval-split test'
                        cmd += f' done: {out_dir / "DONE.json"}'
                        lines.append(cmd)
    return lines


def main():
    (SCRIPTS / 'readout_jobs.txt').write_text('\n'.join(readout_jobs()) + '\n')
    (SCRIPTS / 'finetune_jobs.txt').write_text('\n'.join(finetune_jobs()) + '\n')
    nr = len(readout_jobs()) - 1
    nf = len(finetune_jobs()) - 1
    print(f'[make_jobs] readout_jobs.txt: {nr} jobs; finetune_jobs.txt: {nf} jobs')


if __name__ == '__main__':
    main()
