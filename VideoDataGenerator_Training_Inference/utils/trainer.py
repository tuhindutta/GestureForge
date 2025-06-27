
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class HandDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class GruModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super().__init__()
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = 0.2,
            device = device           
        )
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, output = self.gru(x)
        output = self.linear(output[-1])
        return output
    

