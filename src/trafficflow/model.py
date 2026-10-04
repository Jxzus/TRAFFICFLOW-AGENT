"""Graph WaveNet compacto: convoluciones temporales dilatadas causales + convolución sobre grafo
con matriz de adyacencia adaptativa (Wu et al., 2019; evaluado en el benchmark DL-Traff)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_walk_matrix(adj: np.ndarray) -> np.ndarray:
    d = adj.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    return (adj / d).astype(np.float32)


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_supports: int, order: int = 2, dropout: float = 0.3):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d(c_in * (order * n_supports + 1), c_out, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, supports) -> torch.Tensor:  # x: (B, C, N, T)
        outs = [x]
        for a in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("mn,bcnt->bcmt", a, h)
                outs.append(h)
        return self.dropout(self.mlp(torch.cat(outs, dim=1)))


class GraphWaveNet(nn.Module):
    def __init__(
        self,
        adj: np.ndarray,
        in_dim: int = 4,
        out_len: int = 12,
        res_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 128,
        end_channels: int = 256,
        emb_dim: int = 10,
        dilations=(1, 2, 1, 2, 1, 2),
        dropout: float = 0.3,
    ):
        super().__init__()
        n = adj.shape[0]
        self.register_buffer("p_fwd", torch.from_numpy(random_walk_matrix(adj)))
        self.register_buffer("p_bwd", torch.from_numpy(random_walk_matrix(adj.T.copy())))
        self.e1 = nn.Parameter(torch.randn(n, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(emb_dim, n) * 0.1)
        self.dilations = list(dilations)
        self.start = nn.Conv2d(in_dim, res_channels, kernel_size=1)
        self.filt = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.bn = nn.ModuleList()
        for d in self.dilations:
            self.filt.append(nn.Conv2d(res_channels, dilation_channels, (1, 2), dilation=(1, d)))
            self.gate.append(nn.Conv2d(res_channels, dilation_channels, (1, 2), dilation=(1, d)))
            self.skip.append(nn.Conv2d(dilation_channels, skip_channels, 1))
            self.gcn.append(GraphConv(dilation_channels, res_channels, 3, dropout=dropout))
            self.bn.append(nn.BatchNorm2d(res_channels))
        self.end1 = nn.Conv2d(skip_channels, end_channels, 1)
        self.end2 = nn.Conv2d(end_channels, out_len, 1)

    def supports(self):
        adp = F.softmax(F.relu(self.e1 @ self.e2), dim=1)
        return [self.p_fwd, self.p_bwd, adp]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, C) -> pronóstico normalizado (B, out_len, N)."""
        x = self.start(x.permute(0, 3, 2, 1))  # (B, C, N, T)
        sup = self.supports()
        skip_sum = 0
        for i, d in enumerate(self.dilations):
            res = x
            xp = F.pad(x, (d, 0))  # relleno causal: no se usa información futura
            h = torch.tanh(self.filt[i](xp)) * torch.sigmoid(self.gate[i](xp))
            skip_sum = skip_sum + self.skip[i](h[..., -1:])
            h = self.gcn[i](h, sup)
            x = self.bn[i](h + res)
        out = self.end2(F.relu(self.end1(F.relu(skip_sum))))  # (B, out_len, N, 1)
        return out.squeeze(-1)
