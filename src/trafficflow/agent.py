"""Agente TrafficFlow: Agente basado en modelo con capacidad de aprendizaje (AIMA).

Ciclo explícito de cuatro fases (AIMA Russell & Norvig, Cap. 2):
1. Percepción: Recibe lecturas de sensores de velocidad y marca de tiempo, preprocesando
   y actualizando el estado interno y la memoria de corto/largo plazo.
2. Aprendizaje / Crítica: Compara observaciones reales contra pronósticos pasados a 30 min.
   Calcula la métrica de desempeño (MAPE) y dispara recalibración si hay deriva (> 10 %).
3. Decisión / Razonamiento: Emplea el modelo espacial-temporal interno (Graph WaveNet)
   y el perfil histórico (z-score) para determinar el nivel de congestión del corredor,
   distinguir eventos atípicos de recurrentes y seleccionar el plan de mitigación.
4. Acción: Invoca los actuadores (TMS para fases semafóricas y mensajería a conductores),
   aplicando políticas de estabilidad temporal para evitar oscilaciones.
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .actuator import ActionResult, Assessment, SimulatedTMS, apply_assessment
from .congestion import LEVELS, corridor_level, speed_to_level
from .data import STEP_MIN
from .profile import HistoricalProfile


@dataclass
class Perception:
    """Percepción procesada del entorno captada por los sensores."""
    timestamp: str
    num_sensors: int
    valid_sensors: int
    missing_sensors: int
    mean_speed_observed_mph: float
    min_speed_observed_mph: float
    raw_speeds: np.ndarray
    valid_speeds: np.ndarray

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "num_sensors": self.num_sensors,
            "valid_sensors": self.valid_sensors,
            "missing_sensors": self.missing_sensors,
            "mean_speed_observed_mph": self.mean_speed_observed_mph,
            "min_speed_observed_mph": self.min_speed_observed_mph,
        }


class ForecastTracker:
    """Componente de Crítica (AIMA): Compara cada pronóstico con lo observado cuando llega el dato real."""

    def __init__(self, out_len: int, window: int):
        self.out_len = out_len
        self.pending: dict[int, np.ndarray] = {}
        self.window = [deque(maxlen=window) for _ in range(out_len)]  # (suma de errores relativos, n)
        self.total = np.zeros((out_len, 3))  # suma abs, suma rel, n

    def register(self, step: int, pred: np.ndarray):
        """Registra un pronóstico para su evaluación posterior."""
        self.pending[step] = pred
        for old in [k for k in self.pending if k < step - self.out_len]:
            del self.pending[old]

    def observe(self, step: int, obs: np.ndarray):
        """Recibe la observación real del paso actual y evalúa pronósticos emitidos en pasos previos."""
        valid = ~np.isnan(obs) & (obs > 0)
        for h in range(1, self.out_len + 1):
            pred = self.pending.get(step - h)
            if pred is None or not valid.any():
                continue
            err = np.abs(pred[h - 1][valid] - obs[valid])
            rel = err / obs[valid]
            self.window[h - 1].append((rel.sum(), int(valid.sum())))
            self.total[h - 1] += (err.sum(), rel.sum(), valid.sum())

    def rolling_mape(self, h: int) -> tuple[float, int]:
        """Calcula el MAPE móvil en la ventana reciente para el horizonte h."""
        w = self.window[h - 1]
        n = sum(c for _, c in w)
        return (sum(r for r, _ in w) / n, len(w)) if n else (float("nan"), 0)

    def overall(self) -> dict:
        """Reporta métricas acumuladas globales para todos los horizontes."""
        out = {}
        for h in range(1, self.out_len + 1):
            a, r, n = self.total[h - 1]
            if n:
                out[f"{STEP_MIN * h}min"] = {"MAE": a / n, "MAPE": r / n, "n": int(n)}
        return out


class TrafficFlowAgent:
    """Agente inteligente de control y predicción de tráfico para corredor vial.

    Arquitectura: Agente basado en modelo con capacidad de aprendizaje.
    - Modelo interno: Historial de series temporales, conectividad de sensores (grafo) y perfil histórico.
    - Elemento de crítica: ForecastTracker evalúa el error relativo medio (MAPE).
    - Elemento de aprendizaje: Recalibración adaptativa (fine-tuning) ante deriva conceptual.
    - Elemento de rendimiento: Graph WaveNet + reglas de decisión de control semafórico TMS.
    """

    def __init__(
        self,
        forecaster,
        profile: HistoricalProfile,
        sensor_ids: list[str],
        cfg: dict,
        tms=None,
        in_len: int = 12,
        out_len: int = 12,
        execution_log_path: str | None = None,
    ):
        self.f = forecaster
        self.profile = profile
        self.ids = list(sensor_ids)
        self.cfg = cfg
        self.tms = tms or SimulatedTMS()
        self.in_len = in_len
        self.out_len = out_len
        self.h = cfg["eval_horizon_steps"]  # ej. 6 pasos de 5 min = horizonte de 30 min

        # Estado interno / Memoria de corto y largo plazo (Agente basado en modelo)
        self.buf_v: deque = deque(maxlen=in_len)
        self.buf_t: deque = deque(maxlen=in_len)
        self.hist_v: deque = deque(maxlen=cfg["history_steps"])
        self.hist_t: deque = deque(maxlen=cfg["history_steps"])

        # Componente de Crítica y Aprendizaje continuo
        self.tracker = ForecastTracker(out_len, cfg["drift_window_steps"])
        self.step_i = -1
        self.act_state: dict = {}
        self.last_recal = -10**9

        # Registros históricos y logs de ejecución
        self.assessments: list[Assessment] = []
        self.recalibrations: list[dict] = []
        self.execution_log: list[dict] = []
        self.execution_log_path = execution_log_path

        if self.execution_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.execution_log_path)), exist_ok=True)
            # Inicializar o truncar archivo de log de ejecución para la corrida
            open(self.execution_log_path, "w", encoding="utf-8").close()

    # =========================================================================
    # 1. CICLO DE PERCEPCIÓN (Entradas y Sensores - Sección 5)
    # =========================================================================
    def perceive(self, ts: pd.Timestamp, speeds: np.ndarray) -> Perception:
        """Fase 1: Recibe percepciones de los sensores de velocidad y reloj del sistema.

        Valida la calidad de los datos, trata valores faltantes (0 o NaN) y actualiza
        tanto el estado interno como el componente de crítica con la observación real.
        """
        self.step_i += 1
        arr = np.asarray(speeds, dtype=np.float32)
        valid_mask = (arr > 0) & (~np.isnan(arr))
        obs = np.where(valid_mask, arr, np.nan).astype(np.float32)

        # Informar al crítico de lo que realmente ocurrió para evaluar pronósticos anteriores
        self.tracker.observe(self.step_i, obs)

        # Actualizar memoria interna (estado del modelo)
        self.buf_v.append(obs)
        self.buf_t.append(ts)
        self.hist_v.append(obs)
        self.hist_t.append(ts)

        valid_count = int(valid_mask.sum())
        missing_count = int((~valid_mask).sum())
        mean_speed = float(np.nanmean(obs)) if valid_count > 0 else 0.0
        min_speed = float(np.nanmin(obs)) if valid_count > 0 else 0.0

        return Perception(
            timestamp=str(ts),
            num_sensors=len(self.ids),
            valid_sensors=valid_count,
            missing_sensors=missing_count,
            mean_speed_observed_mph=mean_speed,
            min_speed_observed_mph=min_speed,
            raw_speeds=arr,
            valid_speeds=obs,
        )

    # =========================================================================
    # 2. CICLO DE DECISIÓN / RAZONAMIENTO (Modelo interno y perfil histórico)
    # =========================================================================
    def decide(self, perception: Perception) -> Assessment | None:
        """Fase 2: Razona sobre el estado interno para proyectar el tráfico a 30 minutos.

        Contrasta el pronóstico espacial con el perfil histórico (z-score) para
        determinar si la congestión es atípica (accidente/obra) o recurrente (hora pico),
        y selecciona el plan de mitigación semafórica.
        """
        if len(self.buf_v) < self.in_len:
            # Estado de inicio: esperando acumular el horizonte histórico mínimo
            return None

        idx = pd.DatetimeIndex(list(self.buf_t))
        # Consulta al modelo predictivo interno (Graph WaveNet / baseline)
        pred = self.f.predict(np.stack(self.buf_v), idx)

        # Registrar el nuevo pronóstico para evaluación futura cuando pasen 30 minutos
        self.tracker.register(self.step_i, pred)

        # Evaluación analítica de congestión y atipicidad
        assessment = self._assess(idx[-1], pred)
        self.assessments.append(assessment)
        return assessment

    def _assess(self, now: pd.Timestamp, pred: np.ndarray) -> Assessment:
        """Razonamiento deductivo: calcula nivel del corredor y z-score contra el patrón típico."""
        c, h = self.cfg, self.h
        target = pd.DatetimeIndex(now + pd.to_timedelta(np.arange(1, h + 1) * STEP_MIN, unit="min"))
        mu, sd = self.profile.at(target)

        # Z-score medio del horizonte por sensor contra la memoria histórica
        z = ((pred[:h] - mu) / sd).mean(axis=0)
        atypical = z < -c["atypical_z"]
        frac = float(atypical.mean())

        sensor_speed = pred[:h].mean(axis=0)
        th = c["level_thresholds_mph"]
        level = corridor_level(sensor_speed, th, c["corridor_percentile"])

        # Clasificación del estado
        if frac >= c["atypical_fraction"] and level >= 1:
            status = "atipica"
        elif level >= 2:
            status = "recurrente"
        else:
            status = "normal"

        affected = (
            [self.ids[i] for i in np.where(atypical | (speed_to_level(sensor_speed, th) >= 2))[0]]
            if status != "normal"
            else []
        )

        return Assessment(
            timestamp=str(now),
            status=status,
            level=level,
            level_name=LEVELS[level],
            corridor_speed_mph=float(np.nanpercentile(sensor_speed, c["corridor_percentile"])),
            atypical_fraction=frac,
            affected_sensors=affected,
        )

    # =========================================================================
    # 3. CICLO DE ACCIÓN (Actuadores TMS y Mensajería - Sección 6)
    # =========================================================================
    def act(self, decision: Assessment | None) -> ActionResult:
        """Fase 3: Ejecuta las acciones decididas sobre los actuadores.

        Aplica filtros de histéresis temporal para evitar fluctuaciones rápidas
        en el sistema de control semafórico.
        """
        if decision is None:
            return ActionResult(action_taken=False, timestamp=None)

        return apply_assessment(decision, self.tms, self.act_state)

    # =========================================================================
    # 4. CICLO DE APRENDIZAJE Y CRÍTICA (Adaptación y Recalibración)
    # =========================================================================
    def learn(self) -> dict | None:
        """Fase 4: Elemento de aprendizaje de AIMA.

        Monitorea el MAPE móvil a 30 minutos frente a la meta SMART (< 10 %).
        Si se detecta deriva o degradación por encima del umbral, ejecuta
        un reentrenamiento adaptativo (fine-tuning) sobre el historial reciente.
        """
        c = self.cfg
        mape, n = self.tracker.rolling_mape(self.h)

        learning_event = {
            "step": self.step_i,
            "rolling_mape_30min": float(mape) if not np.isnan(mape) else None,
            "samples": n,
            "meets_target": bool(mape < c["mape_target"]) if not np.isnan(mape) else True,
            "recalibrated": False,
        }

        if n < c["drift_min_samples"] or not (mape > c["mape_target"]):
            return learning_event

        if self.step_i - self.last_recal < c["recalibration_cooldown_steps"]:
            return learning_event

        # Disparar recalibración por deriva detectada
        self.last_recal = self.step_i
        finetune_loss = None
        recalibrated = False

        if hasattr(self.f, "recalibrate") and len(self.hist_v) >= 3 * self.in_len:
            df = pd.DataFrame(np.stack(self.hist_v), index=pd.DatetimeIndex(list(self.hist_t)))
            finetune_loss = self.f.recalibrate(df)
            recalibrated = finetune_loss is not None

        learning_event.update({
            "recalibrated": recalibrated,
            "finetune_loss": finetune_loss,
            "reason": f"Deriva de error: MAPE a 30 min ({mape:.2%}) superó la meta ({c['mape_target']:.0%})",
        })
        self.recalibrations.append(learning_event)
        return learning_event

    # =========================================================================
    # ORQUESTADOR DEL PASO (Percepción -> Aprendizaje -> Decisión -> Acción -> Log)
    # =========================================================================
    def step(self, ts: pd.Timestamp, speeds: np.ndarray) -> Assessment | None:
        """Ejecuta un ciclo completo del agente inteligente para un intervalo de 5 minutos."""
        # 1. Percepción
        perception = self.perceive(ts, speeds)

        # 2. Aprendizaje y Crítica
        learning_event = self.learn()

        # 3. Decisión
        decision = self.decide(perception)

        # 4. Acción
        action_result = self.act(decision)

        # 5. Registro (Log) de ejecución
        self._log_execution_step(perception, decision, action_result, learning_event)

        return decision

    def _log_execution_step(
        self,
        perception: Perception,
        decision: Assessment | None,
        action: ActionResult,
        learning: dict | None,
    ):
        """Registra percepciones, decisiones y acciones con marca de tiempo para auditoría y Fase 3."""
        entry = {
            "step": self.step_i,
            "timestamp": perception.timestamp,
            "perception": perception.to_dict(),
            "decision": asdict(decision) if decision else None,
            "action": action.to_dict(),
            "learning": learning,
        }
        self.execution_log.append(entry)

        if self.execution_log_path:
            with open(self.execution_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_execution_log(self) -> list[dict]:
        """Retorna el historial completo de ejecución del agente."""
        return self.execution_log

    def report(self) -> dict:
        """Genera el reporte integral de desempeño del agente."""
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
            "total_steps_executed": self.step_i + 1,
        }
