"""Agente TrafficFlow: percibe -> pronostica -> evalúa -> actúa -> aprende del resultado real."""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .actuator import Assessment, SimulatedTMS, apply_assessment
from .congestion import LEVELS, corridor_level, speed_to_level
from .data import STEP_MIN
from .profile import HistoricalProfile


class ForecastTracker:
    """Compara cada pronóstico con lo observado cuando llega el dato real."""

    def __init__(self, out_len: int, window: int):
        self.out_len = out_len
        self.pending: dict[int, np.ndarray] = {}
        self.window = [deque(maxlen=window) for _ in range(out_len)]  # (suma de errores relativos, n)
        self.total = np.zeros((out_len, 3))  # suma abs, suma rel, n

    def register(self, step: int, pred: np.ndarray):
        self.pending[step] = pred
        for old in [k for k in self.pending if k < step - self.out_len]:
            del self.pending[old]

    def observe(self, step: int, obs: np.ndarray):
        valid = ~np.isnan(obs) & (obs > 0)
        for h in range(1, self.out_len + 1):
            pred = self.pending.get(step - h)
            if pred is None or not valid.any():
                continue
            err = np.abs(pred[h - 1][valid] - obs[valid])
            rel = err / obs[valid]
            self.window[h - 1].append((rel.sum(), int(valid.sum())))
            self.total[h - 1] += (err.sum(), rel.sum(), valid.sum())

    def rolling_mape(self, h: int):
        w = self.window[h - 1]
        n = sum(c for _, c in w)
        return (sum(r for r, _ in w) / n, len(w)) if n else (float("nan"), 0)

    def overall(self):
        out = {}
        for h in range(1, self.out_len + 1):
            a, r, n = self.total[h - 1]
            if n:
                out[f"{STEP_MIN * h}min"] = {"MAE": a / n, "MAPE": r / n, "n": int(n)}
        return out


class TrafficFlowAgent:
    def __init__(self, forecaster, profile: HistoricalProfile, sensor_ids, cfg: dict, tms=None, in_len: int = 12, out_len: int = 12):
        self.f, self.profile, self.ids, self.cfg = forecaster, profile, list(sensor_ids), cfg
        self.tms = tms or SimulatedTMS()
        self.in_len, self.out_len = in_len, out_len
        self.h = cfg["eval_horizon_steps"]
        self.buf_v: deque = deque(maxlen=in_len)
        self.buf_t: deque = deque(maxlen=in_len)
        self.hist_v: deque = deque(maxlen=cfg["history_steps"])
        self.hist_t: deque = deque(maxlen=cfg["history_steps"])
        self.tracker = ForecastTracker(out_len, cfg["drift_window_steps"])
        self.step_i = -1
        self.act_state: dict = {}
        self.last_recal = -10**9
        self.assessments: list[Assessment] = []
        self.recalibrations: list[dict] = []

    # ---- percepción + decisión ----
    def step(self, ts: pd.Timestamp, speeds: np.ndarray) -> Assessment | None:
        """Procesa la lectura de los sensores (una por cada 5 min). Devuelve la evaluación o None si aún no hay historial."""
        self.step_i += 1
        obs = np.where(np.asarray(speeds, dtype=np.float32) > 0, speeds, np.nan).astype(np.float32)
        self.tracker.observe(self.step_i, obs)
        for buf, val in ((self.buf_v, obs), (self.buf_t, ts), (self.hist_v, obs), (self.hist_t, ts)):
            buf.append(val)
        self._maybe_recalibrate()
        if len(self.buf_v) < self.in_len:
            return None
        idx = pd.DatetimeIndex(list(self.buf_t))
        pred = self.f.predict(np.stack(self.buf_v), idx)
        self.tracker.register(self.step_i, pred)
        a = self._assess(idx[-1], pred)
        apply_assessment(a, self.tms, self.act_state)
        self.assessments.append(a)
        return a

    # ---- razonamiento: nivel + recurrente vs. atípico ----
    def _assess(self, now: pd.Timestamp, pred: np.ndarray) -> Assessment:
        c, h = self.cfg, self.h
        target = pd.DatetimeIndex(now + pd.to_timedelta(np.arange(1, h + 1) * STEP_MIN, unit="min"))
        mu, sd = self.profile.at(target)
        z = ((pred[:h] - mu) / sd).mean(axis=0)  # z-score medio del horizonte, por sensor
        atypical = z < -c["atypical_z"]
        frac = float(atypical.mean())
        sensor_speed = pred[:h].mean(axis=0)
        th = c["level_thresholds_mph"]
        level = corridor_level(sensor_speed, th, c["corridor_percentile"])
        if frac >= c["atypical_fraction"] and level >= 1:
            status = "atipica"
        elif level >= 2:
            status = "recurrente"
        else:
            status = "normal"
        affected = [self.ids[i] for i in np.where(atypical | (speed_to_level(sensor_speed, th) >= 2))[0]] if status != "normal" else []
        return Assessment(
            timestamp=str(now), status=status, level=level, level_name=LEVELS[level],
            corridor_speed_mph=float(np.nanpercentile(sensor_speed, c["corridor_percentile"])),
            atypical_fraction=frac, affected_sensors=affected,
        )

    # ---- aprendizaje: recalibración si el error real supera la meta ----
    def _maybe_recalibrate(self):
        c = self.cfg
        mape, n = self.tracker.rolling_mape(self.h)
        if n < c["drift_min_samples"] or not mape > c["mape_target"]:
            return
        if self.step_i - self.last_recal < c["recalibration_cooldown_steps"]:
            return
        self.last_recal = self.step_i
        event = {"step": self.step_i, "rolling_mape_30min": mape, "recalibrated": False}
        if hasattr(self.f, "recalibrate") and len(self.hist_v) >= 3 * self.in_len:
            df = pd.DataFrame(np.stack(self.hist_v), index=pd.DatetimeIndex(list(self.hist_t)))
            loss = self.f.recalibrate(df)
            event.update(recalibrated=loss is not None, finetune_loss=loss)
        self.recalibrations.append(event)

    def report(self) -> dict:
        counts = {}
        for a in self.assessments:
            counts[a.status] = counts.get(a.status, 0) + 1
        overall = self.tracker.overall()
        m30 = overall.get(f"{STEP_MIN * self.h}min", {}).get("MAPE")
        return {
            "forecast_error": overall,
            "mape_30min": m30,
            "meets_target": (m30 is not None and m30 < self.cfg["mape_target"]),
            "status_counts": counts,
            "tms_actions": len(self.tms.events),
            "recalibrations": self.recalibrations,
        }
