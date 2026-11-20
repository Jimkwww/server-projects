"""Interface contract only, NOT a fake implementation of any named foundation model.
Server fills this using verified original source. Importing file has no side effects.
"""
from torch import nn
class VerifiedExternalEncoder(nn.Module):
    def __init__(self,original_model,iq_to_original,output_to_sequence,metadata):
        super().__init__();self.original_model=original_model
        self.iq_to_original=iq_to_original;self.output_to_sequence=output_to_sequence;self.metadata=metadata
        required={'repository','commit','checkpoint_sha256','preprocessing','output_dim'}
        if not required.issubset(metadata):raise ValueError('Missing provenance for external model')
    def forward(self,iq):
        original=self.original_model(self.iq_to_original(iq))
        sequence=self.output_to_sequence(original)
        if sequence.ndim!=3 or sequence.shape[0]!=iq.shape[0]:raise ValueError('Expected [B,T,D]')
        return {'sequence':sequence}
