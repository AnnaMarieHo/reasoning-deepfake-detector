import torch, torch.nn as nn
from transformers import CLIPModel

class FusionModel(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32",
                 style_dim=512, hidden_dim=512):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.clip.eval();  [p.requires_grad_(False) for p in self.clip.parameters()]
        self.proj_style = nn.Linear(style_dim, 512)
        self.fc = nn.Sequential(
            nn.Linear(512*3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, text, style_vec):
        with torch.no_grad():
            img_emb = self.clip.get_image_features(image)
            txt_emb = self.clip.get_text_features(text)
        z_style = self.proj_style(style_vec)
        fused = torch.cat([img_emb, txt_emb, z_style], dim=-1)
        return self.fc(fused)
