import torch
from torch.utils.data import DataLoader, random_split
from transformers import CLIPProcessor
from models.fusion_model import FusionModel
from models.style_extractor import StyleExtractor
from models.heads import ClassificationHead, StyleClusterHead, ProjectionHead
from utils.custom_dataset import DeepfakeJSONDataset
from utils.train_utils import train_epoch, validate_epoch
from utils.dataset_cached import CachedEmbeddingDataset

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

# Check dataset distribution
labels = dataset.label.numpy()
print(f"Dataset: {len(dataset)} samples | Real: {(labels==1).sum()} | Fake: {(labels==0).sum()}")

train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)

# Training loop
for epoch in range(10):
    loss_val = train_epoch(base_model, heads, train_dl, opt, device)
    acc, auc = validate_epoch(base_model, heads, val_dl, device)
    print(f"Epoch {epoch+1} | Loss {loss_val:.4f} | Acc {acc:.3f} | AUC {auc:.3f}")

torch.save({
    "base": base_model.state_dict(),
    "heads": {k: v.state_dict() for k,v in heads.items()}
}, "checkpoints/multitask.pt")
