from torch.utils.data import Dataset
import torch, numpy as np

class CachedEmbeddingDataset(Dataset):
    def __init__(self, npz_path):
        cache = np.load(npz_path)
        self.img = torch.tensor(cache["image"], dtype=torch.float32)
        self.txt = torch.tensor(cache["text"], dtype=torch.float32)
        self.style = torch.tensor(cache["style"], dtype=torch.float32)
        self.label = torch.tensor(cache["label"], dtype=torch.float32)
        self.cluster = torch.tensor(cache["cluster"], dtype=torch.long)
        self.sim = torch.tensor(cache["similarity"], dtype=torch.float32)

    def __len__(self): return len(self.label)

    def __getitem__(self, idx):
        return (
            self.img[idx],
            self.txt[idx],
            self.style[idx],
            self.label[idx],
            self.sim[idx],
            self.cluster[idx]
        )
