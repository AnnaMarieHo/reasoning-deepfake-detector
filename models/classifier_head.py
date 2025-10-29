import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = F.relu(self.bn(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)