import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder

class MultimodalDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.text = TextEncoder()
        self.cross = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.fc = nn.Linear(512,3)

    def forward(self, images, text):
        v = self.vision(images).unsqueeze(1)
        t = self.text(text).unsqueeze(1)
        fused = self.cross(torch.cat([v,t], dim=1))
        out = self.fc(fused.mean(dim=1))
        return out