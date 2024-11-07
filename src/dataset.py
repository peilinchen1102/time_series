import torch
from torch.utils.data import Dataset

class PTBXL(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return a sample and its corresponding label
        return self.data[idx], self.labels[idx]