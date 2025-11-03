import torch.nn.functional as F
import torch

def bce_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)

def style_centroid_margin(style_vecs, labels, margin=0.5):
    # Encourage fake styles to deviate from real style centroid.
    real_mask = labels == 1
    if real_mask.sum() == 0:
        # No real samples in batch, skip this loss
        return torch.tensor(0.0, device=style_vecs.device, requires_grad=True)
    
    real_mu = style_vecs[real_mask].mean(0, keepdim=True)
    d = torch.cdist(style_vecs, real_mu)
    return torch.clamp(margin - d, min=0).mean()
