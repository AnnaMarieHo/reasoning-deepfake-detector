import torch.nn as nn
import timm

class VisionEncoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True, output_dim=512):
        super().__init__()
        base = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.encoder = base
        self.proj = nn.Linear(base.num_features, output_dim)

    def forward(self, x):
        feats = self.encoder(x)
        return self.proj(feats)
