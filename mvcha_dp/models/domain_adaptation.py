import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (-ctx.lambd * grad_output, None)


class DomainDiscriminator(nn.Module):

    def __init__(self, in_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, lambd=1.0):
        x = GradientReversalFunction.apply(x, lambd)
        return self.net(x)


def discrepancy_loss(logits1, logits2):
    p1 = torch.softmax(logits1, dim=1)
    p2 = torch.softmax(logits2, dim=1)
    return torch.mean(torch.abs(p1 - p2))
