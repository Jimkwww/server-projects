import argparse
from .data import read_manifest,audit
from .engine import load_config
from .model import Classifier,parameter_count
from .utils import save_json,environment


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--config',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    c=load_config(a.config); info=audit(read_manifest(a.manifest))
    params={}
    for v in ('G','S0','S1','S2','S2_concat'):
        cfg=dict(c['model'],variant=v); params[v]=parameter_count(Classifier(cfg))
    save_json(a.output,{'data':info,'environment':environment(),'parameters':params,'config':c,
                       'note':'Machine audit does not verify historical protocol fidelity. Human/agent code audit required.'})
    print(params)

if __name__=='__main__': main()
