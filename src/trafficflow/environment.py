"""Entorno vial (real o simulado) para TrafficFlow-Agent.

Modela el entorno operativo del corredor vial siguiendo las propiedades de la Sección 3
y los sensores de la Sección 5 del documento de diseño (AIMA Russell & Norvig):
- Parcialmente observable: sensores fijos con posibles lecturas faltantes o ruidosas.
- Estocástico: fluctuaciones aleatorias en la velocidad e incidentes imprevistos.
- Secuencial: la congestión tiene inercia temporal y propagación espacial hacia sensores contiguos.
- Dinámico: el tráfico evoluciona en el tiempo independientemente de la deliberación del agente.
- Continuo: velocidades vehiculares continuas (mph), muestreadas a intervalos discretos de 5 min.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import chronological_split, load_adjacency, load_speed_frame
from .synthetic import make_synthetic


class TrafficEnvironment:
    """Simulador / emulador del entorno del corredor vial para el agente."""

    def __init__(
        self,
        df: pd.DataFrame,
        sensor_ids: list[str],
        adj_matrix: np.ndarray,
        stochastic_noise_std: float = 0.0,
        sensor_failure_prob: float = 0.0,
        closed_loop_feedback: bool = False,
    ):
        self.df = df
        self.sensor_ids = list(sensor_ids)
        self.adj_matrix = adj_matrix
        self.stochastic_noise_std = stochastic_noise_std
        self.sensor_failure_prob = sensor_failure_prob
        self.closed_loop_feedback = closed_loop_feedback

        self.current_step = 0
        self.total_steps = len(df)
        self.rng = np.random.default_rng(42)

        # Estado dinámico para simulación con retroalimentación (closed-loop)
        self._current_speed_modifiers = np.zeros(len(self.sensor_ids), dtype=np.float32)

    @classmethod
    def from_dataset(
        cls,
        h5_path: str,
        adj_path: str,
        split_partition: str = "test",
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
        stochastic_noise_std: float = 0.0,
        sensor_failure_prob: float = 0.0,
        closed_loop_feedback: bool = False,
    ) -> "TrafficEnvironment":
        """Crea el entorno a partir de datos reales (ej. METR-LA)."""
        df = load_speed_frame(h5_path)
        sensor_ids, adj = load_adjacency(adj_path)
        tr, va, te = chronological_split(df, split_ratios)
        subset = {"train": tr, "val": va, "test": te}[split_partition]
        return cls(
            df=subset,
            sensor_ids=sensor_ids,
            adj_matrix=adj,
            stochastic_noise_std=stochastic_noise_std,
            sensor_failure_prob=sensor_failure_prob,
            closed_loop_feedback=closed_loop_feedback,
        )

    @classmethod
    def from_synthetic(
        cls,
        n_sensors: int = 20,
        days: int = 28,
        seed: int = 0,
        split_partition: str = "test",
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
        stochastic_noise_std: float = 1.0,
        sensor_failure_prob: float = 0.02,
        closed_loop_feedback: bool = False,
    ) -> "TrafficEnvironment":
        """Crea un entorno completamente sintético y estocástico para pruebas y CI."""
        df, sensor_ids, adj = make_synthetic(n_sensors=n_sensors, days=days, seed=seed)
        tr, va, te = chronological_split(df, split_ratios)
        subset = {"train": tr, "val": va, "test": te}[split_partition]
        return cls(
            df=subset,
            sensor_ids=sensor_ids,
            adj_matrix=adj,
            stochastic_noise_std=stochastic_noise_std,
            sensor_failure_prob=sensor_failure_prob,
            closed_loop_feedback=closed_loop_feedback,
        )

    def reset(self) -> tuple[pd.Timestamp, np.ndarray]:
        """Reinicia el entorno al inicio del conjunto de datos."""
        self.current_step = 0
        self._current_speed_modifiers = np.zeros(len(self.sensor_ids), dtype=np.float32)
        return self._generate_observation(self.current_step)

    def step(self, action: dict | None = None) -> tuple[pd.Timestamp, np.ndarray, bool, dict]:
        """Avanza 1 paso temporal (5 minutos) en el entorno.

        Aplica de forma opcional el efecto de la acción del agente (retroalimentación en lazo cerrado)
        e introduce estocasticidad y fallas aleatorias de sensores.
        """
        if self.current_step >= self.total_steps:
            last_ts = self.df.index[-1]
            return last_ts, np.full(len(self.sensor_ids), np.nan, dtype=np.float32), True, {"done": True}

        # Aplicación de retroalimentación en el entorno si está habilitada
        if self.closed_loop_feedback and action is not None:
            self._apply_action_effects(action)

        ts, speeds = self._generate_observation(self.current_step)
        self.current_step += 1
        done = self.current_step >= self.total_steps
        info = {
            "step": self.current_step - 1,
            "raw_timestamp": ts,
            "closed_loop_active": self.closed_loop_feedback,
        }
        return ts, speeds, done, info

    def _apply_action_effects(self, action: dict):
        """Si el agente aplica un plan de mitigación, mejora moderadamente el flujo en sensores afectados."""
        plan = action.get("plan")
        affected = action.get("sensors", [])
        if not hasattr(self, "_sensor_id_to_idx"):
            self._sensor_id_to_idx = {s: i for i, s in enumerate(self.sensor_ids)}
        if plan in ("PICO_RECURRENTE", "INCIDENTE") and affected:
            bonus = 4.0 if plan == "INCIDENTE" else 2.5
            sensor_indices = [self._sensor_id_to_idx[s] for s in affected if s in self._sensor_id_to_idx]
            if sensor_indices:
                self._current_speed_modifiers[sensor_indices] = bonus
        else:
            # Decaimiento del efecto si se restablece el plan base
            self._current_speed_modifiers *= 0.5

    def _generate_observation(self, step_idx: int) -> tuple[pd.Timestamp, np.ndarray]:
        """Genera la percepción respetando las propiedades del ambiente (ruido, ceros/faltantes)."""
        ts = self.df.index[step_idx]
        base_speeds = self.df.iloc[step_idx].to_numpy(dtype=np.float32, copy=True)

        # Aplicar efecto de retroalimentación
        if self.closed_loop_feedback:
            base_speeds = np.clip(base_speeds + self._current_speed_modifiers, 0.0, 85.0)

        # Propiedad Estocástica: ruido aleatorio en la percepción
        if self.stochastic_noise_std > 0.0:
            noise = self.rng.normal(0.0, self.stochastic_noise_std, size=base_speeds.shape).astype(np.float32)
            valid_mask = base_speeds > 0.0
            base_speeds[valid_mask] = np.clip(base_speeds[valid_mask] + noise[valid_mask], 1.0, 85.0)

        # Propiedad Parcialmente Observable: fallas aleatorias puntuales en la lectura de sensores
        if self.sensor_failure_prob > 0.0:
            failures = self.rng.random(size=base_speeds.shape) < self.sensor_failure_prob
            base_speeds[failures] = 0.0  # en METR-LA 0.0 representa dato faltante/sensor no disponible

        return ts, base_speeds
