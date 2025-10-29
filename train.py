from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset
from models.fusion_module import MultimodalDetector
import torch.nn as nn
import torch
import os
print(os.listdir("data"))
print(os.listdir("data/combined"))


dataset = CustomDataset("data/combined")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)


model = MultimodalDetector().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for images, texts, labels in train_loader:
        images = images.to("cuda")
        labels = labels.long().to("cuda")
        logits = model(images, texts)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/multimodal_detector.pt")
print("Model saved to checkpoints/multimodal_detector.pt")