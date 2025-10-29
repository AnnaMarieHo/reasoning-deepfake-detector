from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, "images")
        self.metadata_path = os.path.join(root_dir, "final_combined_model_input.json")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        self.img_transform = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)
        text = row["text"]
        label = row["label"]
        return image, text, label