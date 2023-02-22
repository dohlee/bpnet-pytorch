import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BPNetDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'wt_emb': self.data[idx][0],
            'mut_emb': self.data[idx][1],
            'label': self.label[idx],
        }