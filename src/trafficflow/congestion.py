"""Clasificación del nivel de congestión a partir de la velocidad (mph)."""
from __future__ import annotations

import numpy as np

LEVELS = ["fluido", "moderado", "denso", "congestionado"]


def speed_to_level(speed, thresholds=(50, 35, 20)):
    """Nivel 0..3 (arreglo o escalar). thresholds: umbrales descendentes para fluido/moderado/denso."""
    s = np.asarray(speed)
    level = np.full(s.shape, 3, dtype=int)
    for lv, th in enumerate(thresholds):
        level = np.where((s >= th) & (level == 3), lv, level)
    return level if level.shape else int(level)


def corridor_level(sensor_speeds: np.ndarray, thresholds=(50, 35, 20), percentile: float = 25.0) -> int:
    """Nivel del corredor: se usa un percentil bajo para que un tramo muy lento no quede diluido."""
    return int(speed_to_level(np.nanpercentile(sensor_speeds, percentile), thresholds))
