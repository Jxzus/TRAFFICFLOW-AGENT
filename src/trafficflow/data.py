"""Carga de METR-LA, escalado y ventanas deslizantes sin fuga de datos."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

STEP_MIN = 5
SLOTS_PER_DAY = 288


def load_speed_frame(path: str) -> pd.DataFrame:
    """metr-la.h5 -> DataFrame (T, N) en mph. Los ceros se tratan como faltantes (NaN)."""
    df = pd.read_hdf(path).astype("float32")
    return df.where(df > 0)


def load_adjacency(path: str):
    """adj_mx.pkl del repo DCRNN -> (sensor_ids, matriz de adyacencia (N, N))."""
    with open(path, "rb") as f:
        sensor_ids, _, adj = pickle.load(f, encoding="latin1")
    return list(sensor_ids), np.asarray(adj, dtype=np.float32)


def chronological_split(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2)):
    n = len(df)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return df.iloc[:n_tr], df.iloc[n_tr : n_tr + n_va], df.iloc[n_tr + n_va :]


@dataclass
class Scaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scaler":
        return cls(float(np.nanmean(values)), float(np.nanstd(values)))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """(T, 2): hora del día en [0,1) y día de la semana / 7."""
    tod = (index.hour * 60 + index.minute) / 1440.0
    dow = index.dayofweek / 7.0
    return np.stack([tod, dow], axis=1).astype(np.float32)


def make_input(speeds: np.ndarray, feats: np.ndarray, scaler: Scaler) -> np.ndarray:
    """speeds (B,T,N) con NaN, feats (B,T,2) -> entrada del modelo (B,T,N,4).

    Canales: velocidad normalizada (faltante=0), máscara de dato válido, hora del día, día de la semana.
    """
    mask = (~np.isnan(speeds)).astype(np.float32)
    s = np.nan_to_num(scaler.transform(speeds), nan=0.0).astype(np.float32)
    n = speeds.shape[2]
    tod = np.repeat(feats[:, :, 0:1], n, axis=2)
    dow = np.repeat(feats[:, :, 1:2], n, axis=2)
    return np.stack([s, mask, tod, dow], axis=-1).astype(np.float32)


class WindowDataset:
    """Ventanas (entrada de in_len pasos -> objetivo de out_len pasos) generadas bajo demanda."""

    def __init__(self, df: pd.DataFrame, scaler: Scaler, in_len: int = 12, out_len: int = 12):
        self.values = df.to_numpy(dtype=np.float32)
        self.feats = time_features(df.index)
        self.index = df.index
        self.scaler, self.in_len, self.out_len = scaler, in_len, out_len

    def __len__(self) -> int:
        return max(0, len(self.values) - self.in_len - self.out_len + 1)

    def get(self, ids):
        ids = np.asarray(ids)
        xi = ids[:, None] + np.arange(self.in_len)
        yi = ids[:, None] + self.in_len + np.arange(self.out_len)
        x = make_input(self.values[xi], self.feats[xi], self.scaler)
        return x, self.values[yi]

    def raw_inputs(self, ids):
        ids = np.asarray(ids)
        xi = ids[:, None] + np.arange(self.in_len)
        return self.values[xi]

    def target_index(self, ids) -> np.ndarray:
        """Posición temporal (en self.index) del primer paso objetivo de cada ventana."""
        return np.asarray(ids) + self.in_len
