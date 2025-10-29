import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/MiniLM-L12-H384-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False #FROZEN
        self.proj = nn.Linear(self.encoder.config.hidden_size, 512)

    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=100
        ).to(next(self.encoder.parameters()).device)
        feats = self.encoder(**inputs).last_hidden_state
        pooled = feats.mean(dim=1)
        return self.proj(pooled)