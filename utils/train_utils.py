import torch
import torch.nn.functional as F
from utils.losses import bce_loss, style_centroid_margin
from utils.metrics import compute_metrics

def train_epoch(base_model, heads, dataloader, optimizer, device):
    """
    One training epoch for cached embeddings.
    dataloader yields: (img_emb, txt_emb, style_vec, label, sim, cluster)
    """
    base_model.train()
    for h in heads.values():
        h.train()

    for img_emb, txt_emb, style, y, sim, cluster in dataloader:
        img_emb, txt_emb = img_emb.to(device), txt_emb.to(device)
        style, y, cluster = style.to(device), y.to(device), cluster.to(device)

        # project style and fuse embeddings
        z_style = base_model.proj_style(style)
        fused = torch.cat([img_emb, txt_emb, z_style], dim=-1)

        #Forward through heads
        logits_main = heads["real_fake"](fused)
        logits_style = heads["style_cluster"](fused)
        proj_img = heads["proj_img"](img_emb)
        proj_txt = heads["proj_txt"](txt_emb)

        # Losses
        loss_main = bce_loss(logits_main, y)
        loss_cluster = F.cross_entropy(logits_style, cluster.clamp_min(0))
        loss_style = style_centroid_margin(style, y)
        align_loss = 1 - F.cosine_similarity(proj_img, proj_txt, dim=-1).mean()

        # Weighted sum
        loss = loss_main + 0.1 * loss_cluster + 0.2 * loss_style + 0.05 * align_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss.item())


def validate_epoch(base_model, heads, dataloader, device):
    base_model.eval()
    for h in heads.values():
        h.eval()

    y_true, y_prob = [], []

    with torch.no_grad():
        for img_emb, txt_emb, style, y, sim, cluster in dataloader:
            img_emb, txt_emb = img_emb.to(device), txt_emb.to(device)
            style, y = style.to(device), y.to(device)

            z_style = base_model.proj_style(style)
            fused = torch.cat([img_emb, txt_emb, z_style], dim=-1)

            probs = torch.sigmoid(heads["real_fake"](fused))
            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc, auc = compute_metrics(y_true, y_prob)
    return acc, auc
