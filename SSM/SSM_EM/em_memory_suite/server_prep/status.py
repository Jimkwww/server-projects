"""One-shot status summary: main experiment (2 GPUs) + historical baseline + GPU usage."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def plan_status(name):
    p = ROOT / 'results' / name / 'plan'
    if not (p / 'status.json').exists():
        return f'{name}: (no status yet)'
    s = json.loads((p / 'status.json').read_text())
    done = sum(1 for v in s.values() if v.get('status') == 'done')
    fail = [k for k, v in s.items() if v.get('status') == 'failed']
    run = [k for k, v in s.items() if v.get('status') == 'running']
    return f'{name}: done={done}/{len(s)} failed={len(fail)} running={run}'


def main():
    for gpu in ['main3_cuda1', 'main3_cuda2']:
        print(plan_status(gpu))
    p0 = ROOT / 'results' / 'main3_cuda0' / 'plan' / 'status.json'
    if p0.exists():
        print(plan_status('main3_cuda0'))
    else:
        print('main3_cuda0: 夜班未启动（GPU0 21:00-09:00 CST 跑）')
    hist = ROOT / 'results' / 'historical'
    if hist.exists():
        js = list(hist.glob('*.json'))
        print(f'historical: {len(js)}/24 results')
        for f in sorted(js):
            d = json.loads(f.read_text())
            print(f'  {d["backbone"]} s{d["seed"]} {d["task"]:14s} {d["mode"]:8s} '
                  f'OA={d["oa"]:.4f} F1={d["macro_f1"]:.4f}')
    # pretrain progress: latest epoch line of any currently-running pretrain run
    for gpu in ['main3_cuda1', 'main3_cuda2', 'main3_cuda0']:
        rd = ROOT / 'results' / gpu / 'plan' / 'runs'
        if not rd.exists():
            continue
        hists = sorted(rd.glob('pre_*/history.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
        if hists:
            lines = hists[0].read_text().splitlines()
            if lines:
                d = json.loads(lines[-1])
                print(f'{gpu} 最近预训练 {hists[0].parent.name}: epoch {d["epoch"]} train_loss {d["train_loss"]:.4f}')
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True)
    print('GPU:\n' + out.stdout.strip())


if __name__ == '__main__':
    main()
