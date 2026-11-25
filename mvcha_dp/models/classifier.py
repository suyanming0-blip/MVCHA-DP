import torch.nn as nn


class DualClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        out1 = self.classifier1(x)
        out2 = self.classifier2(x)
        return (out1, out2)
