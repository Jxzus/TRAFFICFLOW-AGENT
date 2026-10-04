"""Interfaz con el sistema de gestión de tránsito (TMS).

En el piloto se usa SimulatedTMS, que solo registra las acciones. Para conectar un sistema real
basta con implementar TrafficManagementClient (por ejemplo, sobre una API REST del centro de control).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Assessment:
    timestamp: str
    status: str  # "normal" | "recurrente" | "atipica"
    level: int  # 0..3 del corredor
    level_name: str
    corridor_speed_mph: float
    atypical_fraction: float
    affected_sensors: list = field(default_factory=list)


class TrafficManagementClient(Protocol):
    def adjust_signal_timings(self, plan: dict) -> None: ...
    def publish_driver_message(self, message: str, sensors: list) -> None: ...


class SimulatedTMS:
    def __init__(self, log_path: str | None = None):
        self.events: list[dict] = []
        self.log_path = log_path

    def _log(self, ev: dict):
        self.events.append(ev)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def adjust_signal_timings(self, plan: dict) -> None:
        self._log({"type": "signal_plan", **plan})

    def publish_driver_message(self, message: str, sensors: list) -> None:
        self._log({"type": "driver_message", "message": message, "sensors": sensors})


PLANS = {
    "recurrente": {"plan": "PICO_RECURRENTE", "green_extension_pct": 15,
                   "message": "Tráfico denso habitual a esta hora. Tiempo de viaje superior al normal."},
    "atipica": {"plan": "INCIDENTE", "green_extension_pct": 30,
                "message": "Posible incidente en el corredor. Reduzca la velocidad y considere rutas alternas."},
}


def apply_assessment(assessment: Assessment, tms: TrafficManagementClient, state: dict, repeat_every: int = 6) -> bool:
    """Envía acciones al TMS solo cuando cambia la situación (o cada `repeat_every` pasos para confirmar).

    Devuelve True si se envió alguna acción.
    """
    key = (assessment.status, assessment.level)
    state["steps"] = state.get("steps", 0) + 1
    changed = key != state.get("last_key")
    due = assessment.status != "normal" and state["steps"] - state.get("last_sent_step", -10**9) >= repeat_every
    if not (changed or due):
        return False
    state["last_key"], state["last_sent_step"] = key, state["steps"]
    if assessment.status == "normal":
        if state.get("active"):
            tms.adjust_signal_timings({"plan": "RESTABLECER_PLAN_BASE", "timestamp": assessment.timestamp})
            tms.publish_driver_message("Tráfico fluido en el corredor.", [])
            state["active"] = False
            return True
        return False
    spec = PLANS[assessment.status]
    tms.adjust_signal_timings({
        "plan": spec["plan"], "green_extension_pct": spec["green_extension_pct"],
        "level": assessment.level_name, "sensors": assessment.affected_sensors, "timestamp": assessment.timestamp,
    })
    tms.publish_driver_message(spec["message"], assessment.affected_sensors)
    state["active"] = True
    return True
