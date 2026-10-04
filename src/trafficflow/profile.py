"""Perfil histórico por sensor, día de la semana y franja de 5 min (patrón 'típico')."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SLOTS_PER_DAY


def _slots(index: pd.DatetimeIndex):
    return np.asarray(index.dayofweek), np.asarray((index.hour * 60 + index.minute) // 5)


class HistoricalProfile:
    """Media y desviación de la velocidad para cada (día de la semana, franja de 5 min, sensor).

    Se agrupan las franjas vecinas (+-pool) para tener suficientes muestras por celda.
    """

    def __init__(self, pool: int = 2, min_std: float = 2.0):
        self.pool, self.min_std = pool, min_std

    def fit(self, df: pd.DataFrame) -> "HistoricalProfile":
        x = df.to_numpy(dtype=np.float64)
        n = x.shape[1]
        dow, slot = _slots(df.index)
        valid = ~np.isnan(x)
        xz = np.where(valid, x, 0.0)
        s1 = np.zeros((7, SLOTS_PER_DAY, n))
        s2 = np.zeros_like(s1)
        cnt = np.zeros_like(s1)
        np.add.at(s1, (dow, slot), xz)
        np.add.at(s2, (dow, slot), xz**2)
        np.add.at(cnt, (dow, slot), valid.astype(np.float64))
        shifts = range(-self.pool, self.pool + 1)
        s1, s2, cnt = (sum(np.roll(a, k, axis=1) for k in shifts) for a in (s1, s2, cnt))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / cnt
            var = s2 / cnt - mean**2
        std = np.sqrt(np.clip(var, 0, None))
        sensor_mean = np.nanmean(x, axis=0)
        sensor_std = np.nanstd(x, axis=0)
        self.mean = np.where(np.isnan(mean) | (cnt < 3), sensor_mean, mean).astype(np.float32)
        self.std = np.maximum(np.where(np.isnan(std) | (cnt < 3), sensor_std, std), self.min_std).astype(np.float32)
        return self

    def at(self, index: pd.DatetimeIndex):
        """(media, desviación) esperadas para cada instante del índice: arreglos (len(index), N)."""
        dow, slot = _slots(index)
        return self.mean[dow, slot], self.std[dow, slot]
