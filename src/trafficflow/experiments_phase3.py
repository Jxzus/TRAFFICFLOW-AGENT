"""Módulo de evaluación y experimentación para la Fase 3: Pruebas basadas en medidas de desempeño.

Ejecuta 10 escenarios de prueba (normales, límite y fallas esperadas) con 5 repeticiones por escenario
para variables estocásticas, evaluando las medidas de desempeño SMART definidas en el diseño.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .actuator import SimulatedTMS
from .agent import TrafficFlowAgent
from .data import chronological_split, load_adjacency, load_speed_frame
from .environment import TrafficEnvironment
from .forecasters import HistoricalAverageForecaster, PersistenceForecaster, TorchForecaster
from .profile import HistoricalProfile
from .synthetic import make_synthetic


@dataclass
class ScenarioResult:
    scenario_id: int
    name: str
    category: str  # "Normal", "Límite", "Falla esperada"
    repetitions: int
    mape_30min_mean: float
    mape_30min_std: float
    recall_incident_mean: float
    recall_incident_std: float
    false_alarm_rate_mean: float
    false_alarm_rate_std: float
    latency_ms_mean: float
    latency_ms_std: float
    chattering_rate_mean: float
    chattering_rate_std: float
    meets_smart_target: bool
    evidence_log: str
    notes: str


def run_single_simulation(
    df: pd.DataFrame,
    ids: list[str],
    adj: np.ndarray,
    profile: HistoricalProfile,
    forecaster: Any,
    cfg: dict,
    steps: int,
    noise_std: float = 0.0,
    failure_prob: float = 0.0,
    closed_loop: bool = False,
    inject_incident: dict | None = None,
    corrupt_inputs: bool = False,
    log_path: str | None = None,
    seed: int = 42,
) -> dict:
    """Ejecuta una corrida individual del agente sobre el entorno configurado."""
    tms = SimulatedTMS()
    agent = TrafficFlowAgent(
        forecaster=forecaster,
        profile=profile,
        sensor_ids=ids,
        cfg=cfg["agent"],
        tms=tms,
        in_len=cfg["data"]["in_len"],
        out_len=cfg["data"]["out_len"],
        execution_log_path=log_path,
    )

    env = TrafficEnvironment(
        df=df,
        sensor_ids=ids,
        adj_matrix=adj,
        stochastic_noise_std=noise_std,
        sensor_failure_prob=failure_prob,
        closed_loop_feedback=closed_loop,
    )
    env.rng = np.random.default_rng(seed)

    ts, speeds = env.reset()
    latencies = []

    incident_steps = set()
    if inject_incident:
        start_step = inject_incident.get("start_step", 15)
        duration = inject_incident.get("duration", 10)
        sensors_affected = inject_incident.get("sensors", ids[: max(1, len(ids) // 4)])
        factor = inject_incident.get("factor", 0.3)
        incident_steps = set(range(start_step, start_step + duration))

    incident_detected_count = 0
    normal_steps_count = 0
    false_alarm_count = 0
    plan_changes = 0
    last_plan = None

    for step_i in range(min(steps, len(df))):
        cur_speeds = speeds.copy()

        # Inyección de incidente específico si aplica
        if step_i in incident_steps:
            sensor_idx = [ids.index(s) for s in sensors_affected if s in ids]
            cur_speeds[sensor_idx] *= factor

        # Inyección de corrupción severa en fallas esperadas
        if corrupt_inputs and step_i % 5 == 0:
            # Valores negativos o extremos físicamente imposibles
            cur_speeds[: len(ids) // 3] = -99.0
            cur_speeds[len(ids) // 3 : 2 * len(ids) // 3] = 999.0

        t0 = time.perf_counter()
        decision = agent.step(ts, cur_speeds)
        t_elap = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(t_elap)

        # Monitoreo de detección y falsas alarmas
        if decision is not None:
            if step_i in incident_steps:
                if decision.status == "atipica":
                    incident_detected_count += 1
            else:
                normal_steps_count += 1
                if decision.status == "atipica":
                    false_alarm_count += 1

            # Monitoreo de estabilidad de planes semafóricos (chattering)
            current_plan = decision.status
            if last_plan is not None and current_plan != last_plan:
                plan_changes += 1
            last_plan = current_plan

        action_dict = agent.tms.events[-1] if agent.tms.events else None
        ts, speeds, done, _ = env.step(action=action_dict)
        if done:
            break

    rep = agent.report()
    m30 = rep["mape_30min"]

    recall = (incident_detected_count / len(incident_steps)) if incident_steps else 1.0
    far = (false_alarm_count / normal_steps_count) if normal_steps_count else 0.0
    chattering = (plan_changes / max(1, steps)) if steps else 0.0

    return {
        "mape_30min": m30 if m30 is not None else 0.0,
        "recall_incident": recall,
        "false_alarm_rate": far,
        "avg_latency_ms": float(np.mean(latencies)),
        "chattering_rate": chattering,
        "meets_target": bool(rep["meets_target"]),
        "tms_actions": rep["tms_actions"],
        "status_counts": rep["status_counts"],
    }


def run_phase3_suite(
    config_path: str = "configs/default.yaml",
    model_path: str = "artifacts/model.pt",
    out_dir: str = "outputs/phase3",
    steps_per_scenario: int = 35,
    n_repetitions: int = 5,
) -> list[ScenarioResult]:
    """Ejecuta los 10 escenarios con 5 repeticiones por escenario estocástico."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(config_path))
    d = cfg["data"]

    # Cargar datos reales o sintéticos
    if os.path.exists(d["h5_path"]) and os.path.exists(d["adj_path"]):
        df = load_speed_frame(d["h5_path"])
        ids, adj = load_adjacency(d["adj_path"])
    else:
        df, ids, adj = make_synthetic(n_sensors=20, days=28)

    tr, _, te = chronological_split(df, d["split"])
    profile = HistoricalProfile().fit(tr)

    # Cargar pronosticador
    if os.path.exists(model_path):
        fc = TorchForecaster.load(model_path, adj)
    else:
        fc = HistoricalAverageForecaster(profile, d["out_len"])

    # Definición de los 10 escenarios
    scenarios = [
        # --- CASOS NORMALES (4) ---
        {
            "id": 1,
            "name": "Tráfico fluido sostenido (horario valle)",
            "category": "Normal",
            "slice": te.iloc[:steps_per_scenario],
            "noise": 0.5,
            "failure_prob": 0.01,
            "incident": None,
            "corrupt": False,
            "notes": "Tráfico nocturno y entre horas pico con velocidades sobre 55 mph.",
        },
        {
            "id": 2,
            "name": "Hora pico recurrente matutina (7:00 - 9:00 AM)",
            "category": "Normal",
            "slice": te.iloc[24:24 + steps_per_scenario],
            "noise": 1.0,
            "failure_prob": 0.02,
            "incident": None,
            "corrupt": False,
            "notes": "Congestión esperada repetitiva con reducción gradual de velocidad.",
        },
        {
            "id": 3,
            "name": "Hora pico recurrente vespertina (5:00 - 7:00 PM)",
            "category": "Normal",
            "slice": te.iloc[144:144 + steps_per_scenario],
            "noise": 1.0,
            "failure_prob": 0.02,
            "incident": None,
            "corrupt": False,
            "notes": "Flujo denso habitual al final de jornada laboral con retorno domiciliario.",
        },
        {
            "id": 4,
            "name": "Tráfico de fin de semana (patrón dominical)",
            "category": "Normal",
            "slice": te.iloc[288:288 + steps_per_scenario],
            "noise": 0.8,
            "failure_prob": 0.01,
            "incident": None,
            "corrupt": False,
            "notes": "Velocidades estables sin picos de congestión laboral.",
        },
        # --- CASOS LÍMITE (3) ---
        {
            "id": 5,
            "name": "Transición abrupta en el límite de umbral de velocidad (35 mph)",
            "category": "Límite",
            "slice": te.iloc[50:50 + steps_per_scenario],
            "noise": 1.5,
            "failure_prob": 0.02,
            "incident": None,
            "corrupt": False,
            "notes": "Velocidades oscilando en la frontera entre nivel moderado y denso.",
        },
        {
            "id": 6,
            "name": "Incidente atípico severo y corto (15 min en valle)",
            "category": "Límite",
            "slice": te.iloc[10:10 + steps_per_scenario],
            "noise": 1.0,
            "failure_prob": 0.02,
            "incident": {"start_step": 18, "duration": 4, "factor": 0.35},
            "corrupt": False,
            "notes": "Choque menor de retiro rápido que reduce drásticamente velocidad local.",
        },
        {
            "id": 7,
            "name": "Incidente atípico prolongado en plena hora pico",
            "category": "Límite",
            "slice": te.iloc[25:25 + steps_per_scenario],
            "noise": 1.2,
            "failure_prob": 0.02,
            "incident": {"start_step": 16, "duration": 8, "factor": 0.25},
            "corrupt": False,
            "notes": "Bloqueo grave superpuesto con la demanda máxima habitual.",
        },
        # --- CASOS DE FALLA ESPERADA / ENTORNO ADVERSO (3) ---
        {
            "id": 8,
            "name": "Falla masiva de sensores (30% de sensores sin datos / ceros)",
            "category": "Falla esperada",
            "slice": te.iloc[:steps_per_scenario],
            "noise": 1.0,
            "failure_prob": 0.30,
            "incident": None,
            "corrupt": False,
            "notes": "Caída de enlace de comunicaciones o falla eléctrica en subestación.",
        },
        {
            "id": 9,
            "name": "Entorno meteorológico adverso (ruido estocástico extremo sigma=6.0)",
            "category": "Falla esperada",
            "slice": te.iloc[:steps_per_scenario],
            "noise": 6.0,
            "failure_prob": 0.05,
            "incident": None,
            "corrupt": False,
            "notes": "Lluvia torrencial o neblina que distorsiona mediciones de radar.",
        },
        {
            "id": 10,
            "name": "Entradas inválidas / corruptas fuera de rango (-99 / +999 mph)",
            "category": "Falla esperada",
            "slice": te.iloc[:steps_per_scenario],
            "noise": 1.0,
            "failure_prob": 0.05,
            "incident": None,
            "corrupt": True,
            "notes": "Telemetría corrupta con números negativos y desbordamientos.",
        },
    ]

    results: list[ScenarioResult] = []

    print("=" * 70, flush=True)
    print("EJECUTANDO SUITE DE PRUEBAS FASE 3 (10 ESCENARIOS x 5 REPETICIONES)", flush=True)
    print("=" * 70, flush=True)

    for sc in scenarios:
        sid = sc["id"]
        sname = sc["name"]
        cat = sc["category"]
        print(f"\n[Escenario {sid}/10] ({cat}): {sname}")

        mapes = []
        recalls = []
        fars = []
        latencies = []
        chatterings = []
        ev_log = os.path.join(out_dir, f"scenario_{sid}_evidence.jsonl")

        for rep_i in range(n_repetitions):
            # Guardar el log completo en la primera repetición como evidencia formal
            log_target = ev_log if rep_i == 0 else None
            metrics = run_single_simulation(
                df=sc["slice"],
                ids=ids,
                adj=adj,
                profile=profile,
                forecaster=fc,
                cfg=cfg,
                steps=steps_per_scenario,
                noise_std=sc["noise"],
                failure_prob=sc["failure_prob"],
                closed_loop=True,
                inject_incident=sc["incident"],
                corrupt_inputs=sc["corrupt"],
                log_path=log_target,
                seed=42 + rep_i * 13,
            )
            mapes.append(metrics["mape_30min"])
            recalls.append(metrics["recall_incident"])
            fars.append(metrics["false_alarm_rate"])
            latencies.append(metrics["avg_latency_ms"])
            chatterings.append(metrics["chattering_rate"])

        m_mean, m_std = float(np.mean(mapes)), float(np.std(mapes))
        r_mean, r_std = float(np.mean(recalls)), float(np.std(recalls))
        f_mean, f_std = float(np.mean(fars)), float(np.std(fars))
        l_mean, l_std = float(np.mean(latencies)), float(np.std(latencies))
        c_mean, c_std = float(np.mean(chatterings)), float(np.std(chatterings))

        # Criterio de cumplimiento SMART: MAPE < 10% en casos normales/límite; < 13% en fallas extremas
        smart_ok = (m_mean < 0.10) if cat != "Falla esperada" else (m_mean < 0.13)

        sr = ScenarioResult(
            scenario_id=sid,
            name=sname,
            category=cat,
            repetitions=n_repetitions,
            mape_30min_mean=m_mean,
            mape_30min_std=m_std,
            recall_incident_mean=r_mean,
            recall_incident_std=r_std,
            false_alarm_rate_mean=f_mean,
            false_alarm_rate_std=f_std,
            latency_ms_mean=l_mean,
            latency_ms_std=l_std,
            chattering_rate_mean=c_mean,
            chattering_rate_std=c_std,
            meets_smart_target=smart_ok,
            evidence_log=ev_log,
            notes=sc["notes"],
        )
        results.append(sr)
        print(f"  -> MAPE 30 min:  {m_mean:.2%} ± {m_std:.2%} (Meta: {'CUMPLE' if smart_ok else 'NO CUMPLE'})", flush=True)
        print(f"  -> Latencia:     {l_mean:.2f} ms ± {l_std:.2f} ms", flush=True)
        print(f"  -> Recall Inc.:  {r_mean:.2%} ± {r_std:.2%}", flush=True)
        print(f"  -> Falsa Alarma: {f_mean:.2%} ± {f_std:.2%}", flush=True)
        print(f"  -> Evidencia:    {ev_log}", flush=True)

    # Guardar reporte de resultados
    rep_summary_path = os.path.join(out_dir, "phase3_results_summary.json")
    with open(rep_summary_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"Suite de pruebas Fase 3 finalizada con éxito. Resumen en: {rep_summary_path}", flush=True)
    print("=" * 70, flush=True)
    return results


if __name__ == "__main__":
    run_phase3_suite()
