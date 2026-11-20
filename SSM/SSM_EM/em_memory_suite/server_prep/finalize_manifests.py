"""Finalize manifests: add class names, null sample-rate metadata, semantic task name for radar."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

RML2018_CLASSES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                   '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                   '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                   'FM', 'GMSK', 'OQPSK']  # standard Deepsig RML2018.01a ordering
RADCHAR_CLASSES = ['coherent_pulse_train', 'barker_code', 'polyphase_barker_code',
                   'frank_code', 'linear_frequency_modulated']


def load(p):
    return json.loads(Path(p).read_text())


def save(p, d):
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False))


def add_meta(m, classes, task_key='modulation'):
    m.setdefault('sample_rate_hz', None)
    m.setdefault('center_frequency_hz', None)
    if classes:
        m['tasks'][task_key]['classes'] = classes
    return m


# AMC
m = load(DATA / 'rml2018_clsdev/manifest.json')
m = add_meta(m, RML2018_CLASSES)
m['provenance'] += ' sample_rate_hz/center_frequency_hz unavailable (source provides none).'
save(DATA / 'rml2018_clsdev/manifest.json', m)

# F1 pretrain
m = load(DATA / 'rml2018_f1/manifest.json')
m = add_meta(m, RML2018_CLASSES)
m['provenance'] += ' sample_rate_hz/center_frequency_hz unavailable.'
save(DATA / 'rml2018_f1/manifest.json', m)

# RadChar: rename task 'modulation' -> 'radar_waveform'
m = load(DATA / 'radchar_tiny/manifest.json')
task = m['tasks'].pop('modulation')
task['classes'] = RADCHAR_CLASSES
m['tasks'] = {'radar_waveform': task}
for s in m['splits'].values():
    y = s['targets'].pop('modulation')
    s['targets']['radar_waveform'] = y
m = add_meta(m, RADCHAR_CLASSES, task_key='radar_waveform')
m['provenance'] += ' task=radar_waveform (radar waveform classification, not comm modulation).'
save(DATA / 'radchar_tiny/manifest.json', m)

print('[finalize] manifests updated')
