import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from models.fusion_model import FusionModel
from models.style_extractor import StyleExtractor
from models.heads import ClassificationHead, StyleClusterHead, ProjectionHead
from utils.dataset import DeepfakeJSONDataset
from utils.metrics import compute_metrics
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
ckpt = torch.load("checkpoints/multitask.pt", map_location=device)

# Rebuild model + heads
base_model = FusionModel().to(device)
base_model.load_state_dict(ckpt["base"])
heads = {
    "real_fake": ClassificationHead(in_dim=512*3).to(device),
    "style_cluster": StyleClusterHead(in_dim=512*3, num_clusters=40).to(device),
    "proj_img": ProjectionHead(in_dim=512).to(device),
    "proj_txt": ProjectionHead(in_dim=512).to(device),
}
for k in heads:
    heads[k].load_state_dict(ckpt["heads"][k])

# Data
clip_proc = CLIPProcessor.from_pretrained("google/siglip-base-patch16-384")
style_extractor = StyleExtractor(device)
dataset = DeepfakeJSONDataset("data/fake_balanced_filtered/data.json", clip_proc, style_extractor)
dl = DataLoader(dataset, batch_size=8)

# Evaluate real/fake head 
y_true, y_prob = [], []
with torch.no_grad():
    for inputs, style, y, sim, cluster in dl:
        style, y = style.to(device), y.to(device)
        pixel = inputs["pixel_values"].squeeze(1).to(device)
        ids = inputs["input_ids"].squeeze(1).to(device)
        mask = inputs["attention_mask"].squeeze(1).to(device)
        img_emb = base_model.clip.get_image_features(pixel)
        txt_emb = base_model.clip.get_text_features(input_ids=ids, attention_mask=mask)
        z_style = base_model.proj_style(style)
        fused = torch.cat([img_emb, txt_emb, z_style], dim=-1)
        probs = torch.sigmoid(heads["real_fake"](fused))
        y_true.extend(y.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
acc, auc = compute_metrics(y_true, y_prob)
print(f"Real/Fake | Acc {acc:.3f} | AUC {auc:.3f}")

# Evaluate style cluster head 
correct, total = 0, 0
with torch.no_grad():
    for inputs, style, y, sim, cluster in dl:
        style, cluster = style.to(device), cluster.to(device)
        pixel = inputs["pixel_values"].squeeze(1).to(device)
        ids = inputs["input_ids"].squeeze(1).to(device)
        mask = inputs["attention_mask"].squeeze(1).to(device)
        img_emb = base_model.clip.get_image_features(pixel)
        txt_emb = base_model.clip.get_text_features(input_ids=ids, attention_mask=mask)
        z_style = base_model.proj_style(style)
        fused = torch.cat([img_emb, txt_emb, z_style], dim=-1)
        pred = torch.argmax(heads["style_cluster"](fused), dim=1)
        mask_valid = cluster >= 0
        correct += (pred[mask_valid] == cluster[mask_valid]).sum().item()
        total += mask_valid.sum().item()
print(f"Style cluster accuracy: {correct/total:.3f}")
