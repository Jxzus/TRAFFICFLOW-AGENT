"""Pronosticadores intercambiables que usa el agente. Interfaz común:

    predict(speeds (in_len, N) con NaN, index DatetimeIndex de in_len) -> (out_len, N) en mph
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import STEP_MIN, Scaler, WindowDataset, make_input, time_features
from .metrics import masked_mae_loss
from .model import GraphWaveNet
from .profile import HistoricalProfile


class PersistenceForecaster:
    """Línea base: la última velocidad observada se mantiene constante."""

    def __init__(self, out_len: int = 12):
        self.out_len = out_len

    def predict(self, speeds, index):
        last = pd.DataFrame(speeds).ffill().to_numpy()[-1]
        last = np.where(np.isnan(last), np.nanmean(speeds), last)
        return np.tile(last, (self.out_len, 1)).astype(np.float32)


class HistoricalAverageForecaster:
    """Línea base: perfil histórico por día de la semana y franja horaria."""

    def __init__(self, profile: HistoricalProfile, out_len: int = 12):
        self.profile, self.out_len = profile, out_len

    def predict(self, speeds, index):
        target = index[-1] + pd.to_timedelta(np.arange(1, self.out_len + 1) * STEP_MIN, unit="min")
        mean, _ = self.profile.at(pd.DatetimeIndex(target))
        return mean.astype(np.float32)


class TorchForecaster:
    def __init__(self, model: GraphWaveNet, scaler: Scaler, device: str = "cpu"):
        self.model, self.scaler, self.device = model.to(device).eval(), scaler, device

    @classmethod
    def load(cls, path: str, adj: np.ndarray, device: str = "cpu") -> "TorchForecaster":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = GraphWaveNet(adj, out_len=ckpt["out_len"], **ckpt["model_cfg"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, Scaler(**ckpt["scaler"]), device)

    @torch.no_grad()
    def predict(self, speeds, index):
        x = make_input(speeds[None], time_features(index)[None], self.scaler)
        pred = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()[0]
        return self.scaler.inverse(pred).astype(np.float32)

    def recalibrate(self, history: pd.DataFrame, epochs: int = 2, lr: float = 1e-4, batch_size: int = 32):
        """Ajuste fino con el historial reciente (los datos más recientes reflejan el régimen actual)."""
        ds = WindowDataset(history, self.scaler, out_len=self.model.end2.out_channels)
        if len(ds) < batch_size:
            return None
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        last = 0.0
        for _ in range(epochs):
            order = np.random.permutation(len(ds))
            for b in range(0, len(order) - batch_size + 1, batch_size):
                x, y = ds.get(order[b : b + batch_size])
                x, y = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
                loss = masked_mae_loss(self.scaler.inverse(self.model(x)), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                last = float(loss)
        self.model.eval()
        return last
