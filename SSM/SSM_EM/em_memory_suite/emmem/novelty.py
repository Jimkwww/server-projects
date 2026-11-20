import argparse
from .evaluate import novelty
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--normal',required=True)
p.add_argument('--anomaly',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
a=p.parse_args();novelty(a.checkpoint,a.normal,a.anomaly,a.out,a.device)
