import torch
from torch.utils.data import DataLoader, Subset
from transformers import CLIPProcessor
from models.fusion_model import FusionModel
from models.style_extractor import StyleExtractor
from models.heads import ClassificationHead, StyleClusterHead, ProjectionHead
from utils.custom_dataset import DeepfakeJSONDataset
from utils.train_utils import train_epoch, validate_epoch
from utils.dataset_cached import CachedEmbeddingDataset
from utils.balanced_sampler import BalancedBatchSampler
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load components
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# style_extractor = StyleExtractor(device)
base_model = FusionModel(style_dim=771).to(device)

# Define heads
heads = {
    "real_fake": ClassificationHead(in_dim=512*3, hidden_dim=512).to(device),
    "style_cluster": StyleClusterHead(in_dim=512*3, num_clusters=40).to(device),
    "proj_img": ProjectionHead(in_dim=512, proj_dim=256).to(device),
    "proj_txt": ProjectionHead(in_dim=512, proj_dim=256).to(device),
}

# Optimizer
params = list(base_model.proj_style.parameters())
for h in heads.values(): params += list(h.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)

dataset = CachedEmbeddingDataset("openfake-annotation/datasets/combined/cache/fused_embeddings.npz")

# Stratified train/val split to ensure both classes in each split
labels = dataset.label.numpy()
print(f"Dataset: {len(dataset)} samples | Real: {(labels==1).sum()} | Fake: {(labels==0).sum()}")

real_idx = np.where(labels == 1)[0]
fake_idx = np.where(labels == 0)[0]

# Shuffle with seed for reproducibility
np.random.seed(42)
np.random.shuffle(real_idx)
np.random.shuffle(fake_idx)

# 80/20 split for each class
train_real = real_idx[:int(0.8 * len(real_idx))]
val_real = real_idx[int(0.8 * len(real_idx)):]
train_fake = fake_idx[:int(0.8 * len(fake_idx))]
val_fake = fake_idx[int(0.8 * len(fake_idx)):]

train_idx = np.concatenate([train_real, train_fake])
val_idx = np.concatenate([val_real, val_fake])

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_labels = labels[train_idx]
print(f"Train: {len(train_ds)} samples | Real: {(train_labels==1).sum()} | Fake: {(train_labels==0).sum()}")
print(f"Val: {len(val_ds)} samples | Real: {(labels[val_idx]==1).sum()} | Fake: {(labels[val_idx]==0).sum()}")

# Use balanced sampler for training to ensure 50/50 batches
batch_size = 32
train_sampler = BalancedBatchSampler(train_labels, batch_size)
train_dl = DataLoader(train_ds, batch_sampler=train_sampler)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(10):
    loss_val = train_epoch(base_model, heads, train_dl, opt, device)
    acc, auc = validate_epoch(base_model, heads, val_dl, device)
    print(f"Epoch {epoch+1} | Loss {loss_val:.4f} | Acc {acc:.3f} | AUC {auc:.3f}")

torch.save({
    "base": base_model.state_dict(),
    "heads": {k: v.state_dict() for k,v in heads.items()}
}, "checkpoints/multitask.pt")
