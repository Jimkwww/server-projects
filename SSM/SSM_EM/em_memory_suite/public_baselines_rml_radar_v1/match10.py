import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'emind_long_compare_v1'
sys.path.insert(0, str(V1))
sys.path.insert(0, str(V1 / 'EMind'))
sys.path.insert(0, str(V1 / 'wavesfm'))
sys.path.insert(0, str(ROOT.parent))          # em_memory_suite (emmem)
sys.path.insert(0, '/home/user_wzk/wzk/projects/SSM/SSM_L')

from transformers.modeling_utils import PreTrainedModel
if not hasattr(PreTrainedModel, 'get_head_mask'):
    def _ghm(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        return head_mask if head_mask is not None else [None] * num_hidden_layers
    PreTrainedModel.get_head_mask = _ghm

def params(m, skip=()):
    return sum(p.numel() for n, p in m.named_parameters() if not any(s in n for s in skip))

TARGET = 10_000_000
print(f'target ~10M (9.0-11.0M)')
for L, nc in [(1024, 24), (512, 5)]:
    print(f'\n=== L={L} n_classes={nc} ===')
    from models.configuration_em import SiTMAEConfig
    from models.modeling_em import SiTMAEForClassification
    for h, inter in [(256,1024),(288,1152),(256,768),(240,960)]:
        cfg = SiTMAEConfig(hidden_size=h, num_hidden_layers=12, num_attention_heads=8,
                           intermediate_size=inter, max_seq_len=(L,1), patch_size=(4,1),
                           num_channels=2, num_labels=nc, attention_type='sdpa', use_fs=False)
        m = SiTMAEForClassification(cfg)
        print(f'  EMind h={h} inter={inter}: enc={params(m,("classifier","")):,}')
    from emmem.models import Encoder
    for dim, depth in [(128,12),(144,12),(152,12),(160,10),(144,10)]:
        cfg = dict(patch=8, dim=dim, states=dim//2, depth=depth, branched=True, physical_input=True,
                   roles=['energy','coherence','spectrum','period'], kind='ssm', aux_route='local',
                   residual=True, selective=True, real=False, scan='parallel')
        m = Encoder(cfg)
        aux = params(m, ())
        # subtract aux/logvar/reconstruct params
        skip_aux = sum(p.numel() for n,p in m.named_parameters() if any(s in n for s in ('aux','logvar','reconstruct')))
        print(f'  structured dim={dim} depth={depth}: enc={aux-skip_aux:,}')
    from models_vit import ModalityAdapterViT
    for emb, depth in [(256,8),(288,8),(288,10),(320,8)]:
        m = ModalityAdapterViT(modality='iq', num_outputs=nc, embed_dim=emb, depth=depth, num_heads=8,
                               mlp_ratio=4, iq_segment_len=16, iq_hop=16, iq_max_tokens=L//16,
                               iq_max_antennas=1, global_pool='token')
        print(f'  WavesFM embed={emb} depth={depth}: enc={params(m,("head","")):,}')
    from pretrain.model import Encoder as QEnc
    for dim, depth, state in [(256,12,128),(288,10,144),(288,12,144),(256,14,128),(272,12,136)]:
        cfg = dict(patch=8, dim=dim, depth=depth, heads=4, state=state, selector_hidden=64, classes=nc,
                   input_length=L, ffn_ratio=4, dropout=0.0, position='none', direction='forward',
                   scan='parallel', ssm_gain=1.0)
        m = QEnc(cfg, 'Q', seed=17)
        print(f'  Q dim={dim} depth={depth} state={state}: {sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters()):,}')
