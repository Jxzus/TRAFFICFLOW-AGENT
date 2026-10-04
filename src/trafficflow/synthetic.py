"""Generador de datos sintéticos con la misma forma que METR-LA (para pruebas y CI)."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd


def make_synthetic(n_sensors: int = 20, days: int = 28, seed: int = 0, start: str = "2012-03-05"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 288, freq="5min")
    tod = np.asarray(idx.hour * 60 + idx.minute) / 60.0
    weekday = np.asarray(idx.dayofweek < 5)
    rush = (
        np.exp(-((tod - 8) ** 2) / 1.5) + 1.2 * np.exp(-((tod - 17.5) ** 2) / 2.0)
    ) * weekday
    base = 65.0
    depth = rng.uniform(15, 35, n_sensors)  # intensidad de la hora pico por sensor
    speeds = base - rush[:, None] * depth[None, :]
    # propagación espacial: cada sensor recibe parte del retraso del sensor anterior
    for j in range(1, n_sensors):
        speeds[1:, j] -= 0.15 * (base - speeds[:-1, j - 1])
    # incidentes aleatorios (caída fuerte y corta en un tramo contiguo de sensores)
    for _ in range(days):
        t0 = rng.integers(0, len(idx) - 40)
        s0 = rng.integers(0, max(1, n_sensors - 4))
        dur = rng.integers(8, 30)
        speeds[t0 : t0 + dur, s0 : s0 + 4] *= rng.uniform(0.35, 0.6)
    speeds += rng.normal(0, 2.0, speeds.shape)
    speeds = np.clip(speeds, 5, 80)
    speeds[rng.random(speeds.shape) < 0.04] = 0.0  # faltantes codificados como 0, como en METR-LA
    cols = [str(773869 + i) for i in range(n_sensors)]
    df = pd.DataFrame(speeds.astype("float32"), index=idx, columns=cols)
    pos = np.arange(n_sensors, dtype=np.float32)
    dist = np.abs(pos[:, None] - pos[None, :])
    adj = np.exp(-(dist**2) / 4.0)
    adj[adj < 0.1] = 0
    return df, cols, adj.astype(np.float32)


def save_synthetic(h5_path: str, adj_path: str, **kw):
    df, cols, adj = make_synthetic(**kw)
    df.to_hdf(h5_path, key="df", mode="w")
    with open(adj_path, "wb") as f:
        pickle.dump((cols, {c: i for i, c in enumerate(cols)}, adj), f)
    return df, cols, adj
