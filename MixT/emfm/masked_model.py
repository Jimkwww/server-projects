"""Position-preserving masked IQ reconstruction (not dropped-token MAE)."""
import torch
from torch import nn
from torch.nn import functional as F
from .models import IQEncoder


def make_patch_mask(valid, patch_size, ratio, generator=None, sample_ids=None, seed=2024):
    if not 0 < ratio < 1:
        raise ValueError('Mask ratio must be strictly between zero and one')
    valid = valid.detach().cpu()
    padded = F.pad(valid, (0, (-valid.shape[1]) % patch_size), value=False)
    patches = padded.reshape(len(valid), -1, patch_size).any(-1)
    result = torch.zeros_like(patches)
    if sample_ids is not None and len(sample_ids) != len(valid):
        raise ValueError('sample_ids batch mismatch')
    for i in range(len(valid)):
        available = patches[i].nonzero().flatten()
        n = len(available)
        if n < 2:
            raise ValueError('Masked reconstruction needs at least two valid patches')
        k = max(1, min(n - 1, int(n * ratio)))
        rng = generator
        if sample_ids is not None:
            rng = torch.Generator().manual_seed((seed + int(sample_ids[i]) * 1000003) % (2 ** 63 - 1))
        chosen = available[torch.randperm(n, generator=rng)[:k]]
        result[i, chosen] = True
    return result


class MaskedIQModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.encoder = IQEncoder(**model_config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_config['dims'][0]))
        nn.init.normal_(self.mask_token, std=.02)
        # Each final token reconstructs its own contiguous time block. Same
        # head rule for both encoders; no unmasked IQ skip connections.
        self.stride = self.encoder.patch_size * 2 ** len(self.encoder.merges)
        self.reconstruction = nn.Linear(self.encoder.output_dim, self.stride * 2)

    def forward(self, iq, valid, patch_mask):
        features, _ = self.encoder(iq, valid, patch_mask, self.mask_token)
        reconstruction = self.reconstruction(features).reshape(len(iq), -1, 2)
        return reconstruction[:, :iq.shape[1]]


def masked_error(prediction, target, valid, patch_mask, patch_size):
    if prediction.shape != target.shape:
        raise ValueError('Reconstruction shape mismatch')
    points = patch_mask.repeat_interleave(patch_size, dim=1)[:, :target.shape[1]] & valid
    count = points.sum() * target.shape[-1]
    if count.item() == 0:
        raise ValueError('No masked valid IQ components')
    # Index first so invalid/unselected positions never enter the loss.
    error = (prediction.float()[points] - target.float()[points]).square().sum()
    zero_error = target.float()[points].square().sum()
    return error, count, zero_error


def load_pretrained_encoder(classifier, checkpoint, model_config, normalization, cls_digest):
    if checkpoint.get('format') != 'emfm_masked_iq_v1':
        raise ValueError('Expected this project masked-pretraining checkpoint')
    meta = checkpoint['metadata']
    if meta['config']['model'] != model_config or meta['config']['data']['normalization'] != normalization:
        raise ValueError('Pretraining/downstream architecture or normalization mismatch')
    if meta['classification_manifest_sha256'] != cls_digest:
        raise ValueError('Pretraining exclusions belong to a different classification split')
    if meta['stage'] != 'dev':
        raise ValueError('Smoke weights are diagnostic; use dev pretraining for transfer')
    # Exact strict load: head/mask token are intentionally not transferred.
    classifier.encoder.load_state_dict(checkpoint['encoder'], strict=True)
