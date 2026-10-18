"""Interfaz con el sistema de gestión de tránsito (TMS) y paneles de información.

En el piloto se usa SimulatedTMS, que registra las acciones y emula el comportamiento
del centro de control. Para conectar un sistema real basta con implementar
TrafficManagementClient (por ejemplo, sobre una API REST del centro de control de tránsito).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Assessment:
    """Evaluación y decisión generada por el agente para el corredor vial."""
    timestamp: str
    status: str  # "normal" | "recurrente" | "atipica"
    level: int  # 0..3 del corredor
    level_name: str
    corridor_speed_mph: float
    atypical_fraction: float
    affected_sensors: list = field(default_factory=list)


@dataclass
class ActionResult:
    """Resultado de la ejecución de una acción en los actuadores."""
    action_taken: bool
    plan: str | None = None
    green_extension_pct: int | None = None
    message: str | None = None
    sensors: list = field(default_factory=list)
    timestamp: str | None = None

    def __bool__(self) -> bool:
        return self.action_taken

    def to_dict(self) -> dict:
        return asdict(self)


class TrafficManagementClient(Protocol):
    """Protocolo de actuador para el sistema de gestión de tránsito."""
    def adjust_signal_timings(self, plan: dict) -> None: ...
    def publish_driver_message(self, message: str, sensors: list) -> None: ...


class SimulatedTMS:
    """Actuador simulado que almacena y audita todas las órdenes emitidas por el agente."""

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
    "recurrente": {
        "plan": "PICO_RECURRENTE",
        "green_extension_pct": 15,
        "message": "Tráfico denso habitual a esta hora. Tiempo de viaje superior al normal.",
    },
    "atipica": {
        "plan": "INCIDENTE",
        "green_extension_pct": 30,
        "message": "Posible incidente en el corredor. Reduzca la velocidad y considere rutas alternas.",
    },
}


def apply_assessment(
    assessment: Assessment,
    tms: TrafficManagementClient,
    state: dict,
    repeat_every: int = 6,
) -> ActionResult:
    """Envía acciones al TMS solo cuando cambia la situación (o cada `repeat_every` pasos para confirmar).

    Aplica políticas de estabilización y enfriamiento para evitar oscilaciones innecesarias
    en los ciclos de los semáforos. Retorna un objeto ActionResult (compatible con bool).
    """
    key = (assessment.status, assessment.level)
    state["steps"] = state.get("steps", 0) + 1
    changed = key != state.get("last_key")
    due = assessment.status != "normal" and state["steps"] - state.get("last_sent_step", -10**9) >= repeat_every

    if not (changed or due):
        return ActionResult(action_taken=False, timestamp=assessment.timestamp)

    state["last_key"], state["last_sent_step"] = key, state["steps"]

    if assessment.status == "normal":
        if state.get("active"):
            tms.adjust_signal_timings({"plan": "RESTABLECER_PLAN_BASE", "timestamp": assessment.timestamp})
            msg = "Tráfico fluido en el corredor."
            tms.publish_driver_message(msg, [])
            state["active"] = False
            return ActionResult(
                action_taken=True,
                plan="RESTABLECER_PLAN_BASE",
                green_extension_pct=0,
                message=msg,
                sensors=[],
                timestamp=assessment.timestamp,
            )
        return ActionResult(action_taken=False, timestamp=assessment.timestamp)

    spec = PLANS[assessment.status]
    plan_info = {
        "plan": spec["plan"],
        "green_extension_pct": spec["green_extension_pct"],
        "level": assessment.level_name,
        "sensors": assessment.affected_sensors,
        "timestamp": assessment.timestamp,
    }
    tms.adjust_signal_timings(plan_info)
    tms.publish_driver_message(spec["message"], assessment.affected_sensors)
    state["active"] = True

    return ActionResult(
        action_taken=True,
        plan=spec["plan"],
        green_extension_pct=spec["green_extension_pct"],
        message=spec["message"],
        sensors=assessment.affected_sensors,
        timestamp=assessment.timestamp,
    )
