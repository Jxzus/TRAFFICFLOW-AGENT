"""Métricas enmascaradas: los valores faltantes (NaN o 0 en METR-LA) no cuentan."""
from __future__ import annotations

import numpy as np
import torch


def _valid(y: np.ndarray) -> np.ndarray:
    return ~np.isnan(y) & (y > 0)


def masked_mae(pred: np.ndarray, y: np.ndarray) -> float:
    m = _valid(y)
    return float(np.abs(pred[m] - y[m]).mean())


def masked_rmse(pred: np.ndarray, y: np.ndarray) -> float:
    m = _valid(y)
    return float(np.sqrt(((pred[m] - y[m]) ** 2).mean()))


def masked_mape(pred: np.ndarray, y: np.ndarray) -> float:
    """Error porcentual absoluto medio (fracción, 0.10 = 10 %)."""
    m = _valid(y)
    return float((np.abs(pred[m] - y[m]) / y[m]).mean())


def horizon_report(pred: np.ndarray, y: np.ndarray, horizons=(3, 6, 12)) -> dict:
    """pred, y: (S, H, N). horizons en pasos de 5 min (3=15 min, 6=30 min, 12=60 min)."""
    out = {}
    for h in horizons:
        p, t = pred[:, h - 1], y[:, h - 1]
        out[f"{5 * h}min"] = {
            "MAE": masked_mae(p, t),
            "RMSE": masked_rmse(p, t),
            "MAPE": masked_mape(p, t),
        }
    return out


def masked_mae_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pérdida L1 enmascarada para entrenamiento (y con NaN = faltante)."""
    mask = (~torch.isnan(y)) & (y > 0)
    y0 = torch.where(mask, y, torch.zeros_like(y))
    loss = torch.abs(pred - y0) * mask
    return loss.sum() / mask.sum().clamp(min=1)
