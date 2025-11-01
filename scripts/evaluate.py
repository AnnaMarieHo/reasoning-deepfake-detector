import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from models.fusion_model import FusionModel
from utils.dataset import DeepfakeJSONDataset
from models.style_extractor import StyleExtractor
from transformers import CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FusionModel().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()

clip_proc = CLIPProcessor.from_pretrained("google/siglip-base-patch16-384")
style_extractor = StyleExtractor(device)
dataset = DeepfakeJSONDataset("data/fake_balanced_filtered/data.json", clip_proc, style_extractor)
dl = DataLoader(dataset, batch_size=8)

# Real/Fake accuracy + AUC
acc, auc = compute_metrics(y_true, y_prob)

# Style Cluster accuracy
correct, total = 0, 0
with torch.no_grad():
    for inputs, style, y, sim, cluster in val_dl:
        style, cluster = style.to(device), cluster.to(device)
        pixel = inputs["pixel_values"].squeeze(1).to(device)
        ids = inputs["input_ids"].squeeze(1).to(device)
        mask = inputs["attention_mask"].squeeze(1).to(device)
        img_emb = base_model.clip.get_image_features(pixel)
        txt_emb = base_model.clip.get_text_features(input_ids=ids, attention_mask=mask)
        z_style = base_model.proj_style(style)
        fused = torch.cat([img_emb, txt_emb, z_style], dim=-1)
        pred = torch.argmax(style_cluster_head(fused), dim=1)
        mask_valid = cluster >= 0
        correct += (pred[mask_valid] == cluster[mask_valid]).sum().item()
        total += mask_valid.sum().item()
print(f"Style cluster accuracy: {correct/total:.3f}")
