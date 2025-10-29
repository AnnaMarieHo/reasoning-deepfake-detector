import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from models.fusion_module import MultimodalDetector
from sklearn.metrics import classification_report, confusion_matrix


dataset = CustomDataset("data/combined")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = MultimodalDetector().to("cuda")
model.load_state_dict(torch.load("checkpoints/multimodal_detector.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, texts, labels in val_loader:
        images = images.to("cuda")
        labels = labels.long().to("cuda")

        logits = model(images, texts)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["real", "full_synthetic", "tampered"]))


train_imgs = {dataset.data[i]['image'] for i in train_dataset.indices}
val_imgs = {dataset.data[i]['image'] for i in val_dataset.indices}
print("Overlap of samples between train and validatio: "len(train_imgs & val_imgs))
