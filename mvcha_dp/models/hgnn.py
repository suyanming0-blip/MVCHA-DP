# Hypergraph encoder and classification heads.

import torch
from torch import nn
import torch.nn.functional as F


class WeightedHypergraphConv(nn.Module):

    def __init__(self, in_ch, out_ch, bias=True):
        super(WeightedHypergraphConv, self).__init__()
        self.theta = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, x, H, W):
        device = x.device
        H = H.to(device).float()
        weights = (W if W.dim() == 1 else W.diag()).to(device).float()
        DV = torch.sum(H * weights, dim=1)
        DV = torch.clamp(DV, min=1e-08)
        DV_inv_sqrt = torch.pow(DV, -0.5)
        DE = torch.sum(H, dim=0)
        DE = torch.clamp(DE, min=1e-08)
        DE_inv = torch.pow(DE, -1.0)
        x = self.theta(x)
        G = DV_inv_sqrt[:, None] * H * (weights * DE_inv) @ H.t() * DV_inv_sqrt[None, :]
        x = G @ x
        return x


class HGNN(nn.Module):

    def __init__(self, in_ch, n_hid, dropout=0.5, *, profile):
        super(HGNN, self).__init__()
        self.dropout = dropout
        if profile not in ('WPDP', 'CVDP', 'CPDP'):
            raise ValueError('Unknown encoder profile')
        self.profile = profile
        self.hgc1 = WeightedHypergraphConv(in_ch, n_hid)
        self.hgc2 = None if profile == 'WPDP' else WeightedHypergraphConv(n_hid, n_hid)
        self.linear1 = nn.Linear(n_hid, 32)
        self.linear2 = nn.Linear(32, 2)

    def encode(self, x, H, W):
        x = F.relu(self.hgc1(x, H, W))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.profile == 'WPDP':
            return x
        return self.hgc2(x, H, W)

    def encoder_parameters(self):
        return list(self.hgc1.parameters()) + ([] if self.hgc2 is None else list(self.hgc2.parameters()))

    def forward(self, x, H, W):
        x_embedding = self.encode(x, H, W)
        x = F.relu(self.linear1(x_embedding))
        x = self.linear2(x)
        return (x, x_embedding)
