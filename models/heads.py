import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic classification head: main "real vs fake" prediction

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=1536, hidden_dim=512, dropout=0.3, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Projection head: contrastive / alignment training
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        # Normalize to unit length for cosine similarity
        return F.normalize(self.proj(x), dim=-1)


# Auxiliary head: predicts style cluster IDs
class StyleClusterHead(nn.Module):
    def __init__(self, in_dim=512, num_clusters=40):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_clusters)
        )

    def forward(self, x):
        return self.classifier(x)


# Multi-task head: joint real/fake + style prediction
class MultiTaskHead(nn.Module):
    def __init__(self, in_dim=1536, hidden_dim=512, num_clusters=40):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.real_fake = nn.Linear(hidden_dim, 1)
        self.cluster_pred = nn.Linear(hidden_dim, num_clusters)

    def forward(self, x):
        shared = self.shared(x)
        return self.real_fake(shared), self.cluster_pred(shared)
