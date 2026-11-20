"""Server-only model shape/parameter checks plus optional real-data smoke loader."""
import argparse
import json
from pathlib import Path
import torch
from .data import RML2018A, collate, sha256
from .models import Classifier
from .train import loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5')
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = {'models': {}, 'data': None}
    config_dir = Path(__file__).parent / 'configs'
    for file in (config_dir / name for name in ('global_small.json', 'global_matched.json', 'swin_small.json')):
        cfg = json.loads(file.read_text())
        model = Classifier(cfg['model'], cfg['classes']).eval()
        x, valid = torch.randn(2, 1024, 2), torch.ones(2, 1024, dtype=torch.bool)
        logits = model(x, valid)
        logits.square().mean().backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError('Nonfinite or missing gradients')
        report['models'][file.stem] = {'parameters': sum(p.numel() for p in model.parameters()),
                                      'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
                                      'logits_shape': list(logits.shape)}
    g = report['models']['global_matched']['parameters']
    s = report['models']['swin_small']['parameters']
    report['matched_relative_gap_to_swin'] = abs(g - s) / s
    if abs(g - s) / s > .05:
        raise AssertionError('Parameter matching exceeds 5%')
    if bool(a.h5) != bool(a.manifest):
        p.error('Supply both --h5 and --manifest')
    if a.h5:
        expected = cfg['data']['manifest_sha256']
        if sha256(a.manifest) != expected:
            raise ValueError('Manifest hash mismatch')
        observed = []
        for workers in (0, 2):
            ds = RML2018A(a.h5, a.manifest, 'smoke', 'train')
            batches = list(loader(ds, 32, workers, True, torch.Generator().manual_seed(2024)))
            result = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
            ids = result['sample_id']
            if len(ids.unique()) != len(ds):
                raise AssertionError('Lost/repeated sample IDs')
            order = ids.argsort()
            observed.append({k: v[order] for k, v in result.items()})
            ds.close()
        for key in observed[0]:
            torch.testing.assert_close(observed[0][key], observed[1][key])
        report['data'] = {'smoke_samples': len(observed[0]['sample_id']), 'workers_0_2_all_fields_equal': True,
                          'manifest_sha256': sha256(a.manifest)}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
