"""Generate round6 job lists (readout + finetune) and the pretrain task matrix.

Round6 extension: 7 group-seed combos x 3 seeds = 21 pretrain runs. Downstream jobs use the
@@GPU@@ token so dispatch.py --gpus 3 spreads them across three A100s (3 workers, one per GPU).

Writes round6_objective_bidirectional/scripts/readout_jobs.txt (42) and finetune_jobs.txt (42).
Pretrain is driven by run_pretrain.sh (see that script), not a dispatch list.

Reused (NOT re-run) this round: round5 G-MAE and Q-MAE downstream results (30 readout + 30
finetune already exist under round5_epoch_scale/runs/downstream/).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R6 = ROOT / 'round6_objective_bidirectional'
R5 = ROOT / 'round5_epoch_scale'
SCRIPTS = R6 / 'scripts'
BACKBONES = ('G', 'Q', 'Q_bi')
SEEDS = (17, 42, 2026)

# round6 task matrix: 7 NEW groups. (G/Q MAE reused from round5.)
PRETRAIN_GROUPS = [
    ('G', 'denoise'),
    ('G', 'joint'),
    ('Q', 'denoise'),
    ('Q', 'joint'),
    ('Q_bi', 'mae'),
    ('Q_bi', 'denoise'),
    ('Q_bi', 'joint'),
]

TASK_MANIFEST = {
    'rml': ROOT / 'configs' / 'data.json',
    'radar': ROOT / 'round2_overnight' / 'configs' / 'radchar.json',
}
TASK_CONFIG = {
    'rml': R6 / 'configs' / 'downstream_nocfo_rml.json',
    'radar': R6 / 'configs' / 'downstream_nocfo_radar.json',
}
F1_MANIFEST = ROOT / 'round3_scale_transfer' / 'configs' / 'manifest_f1.json'
PRETRAIN_CFG = R6 / 'configs' / 'pretrain.json'
R6_SHARED = R6 / 'runs' / 'pretrain' / 'f1' / 'shared_init'
R5_SHARED = R5 / 'runs' / 'pretrain' / 'f1' / 'shared_init'


def run_name(backbone, task, seed):
    return f'{backbone}_{task}_seed{seed}'


def readout_jobs():
    lines = ['# round6 frozen readout jobs (21 encoders x 2 tasks = 42), @@GPU@@ dispatched']
    for bb, task_pt in PRETRAIN_GROUPS:
        for seed in SEEDS:
            run_dir = R6 / 'runs' / 'pretrain' / 'f1' / run_name(bb, task_pt, seed)
            ckpt = run_dir / 'encoder_final.pt'
            for task in ('rml', 'radar'):
                out_json = (R6 / 'runs' / 'downstream' / 'readout' / 'f1' /
                            f'{run_name(bb, task_pt, seed)}__epoch50__{task}.json')
                cmd = (
                    f'python {SCRIPTS / "readout_r6.py"} '
                    f'--checkpoint {ckpt} --backbone {bb} --corpus f1 --seed {seed} '
                    f'--ckpt-name epoch50 --task {task} --device cuda:@@GPU@@ '
                    f'--cache-dir {R6 / "runs" / "downstream" / "features"} '
                    f'--pred-dir {R6 / "runs" / "downstream" / "predictions"} '
                    f'--out-json {out_json} done: {out_json}'
                )
                lines.append(cmd)
    return lines


def finetune_jobs():
    lines = ['# round6 finetune jobs (21 encoders x 2 tasks = 42), @@GPU@@ dispatched']
    for bb, task_pt in PRETRAIN_GROUPS:
        for seed in SEEDS:
            pretrained = R6 / 'runs' / 'pretrain' / 'f1' / run_name(bb, task_pt, seed) / 'encoder_final.pt'
            for task in ('rml', 'radar'):
                out_dir = (R6 / 'runs' / 'downstream' / 'finetune_50ep' / task /
                           f'f1_{run_name(bb, task_pt, seed)}')
                cmd = (
                    f'python {SCRIPTS / "finetune_r6.py"} '
                    f'--config {TASK_CONFIG[task]} --manifest {TASK_MANIFEST[task]} '
                    f'--backbone {bb} --pretrained {pretrained} --out {out_dir} --seed {seed} '
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
    print(f'[make_jobs] pretrain groups: {len(PRETRAIN_GROUPS)} x {len(SEEDS)} seeds = '
          f'{len(PRETRAIN_GROUPS) * len(SEEDS)} runs')


if __name__ == '__main__':
    main()
