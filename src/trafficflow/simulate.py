"""Reproduce el entorno del corredor vial como flujo en vivo a través de TrafficFlow-Agent.

Genera las percepciones respetando las propiedades del entorno (Sección 3 y 5) y registra
cada paso (percepción, decisión, acción y aprendizaje) con marca de tiempo en el log de ejecución.
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from .actuator import SimulatedTMS
from .agent import TrafficFlowAgent
from .data import chronological_split, load_adjacency, load_speed_frame
from .environment import TrafficEnvironment
from .forecasters import HistoricalAverageForecaster, PersistenceForecaster, TorchForecaster
from .profile import HistoricalProfile
from .synthetic import make_synthetic


def main():
    ap = argparse.ArgumentParser(description="Simulador de flujo en vivo para TrafficFlow-Agent")
    ap.add_argument("--config", default="configs/default.yaml", help="Ruta al archivo YAML de configuración")
    ap.add_argument("--model", default="artifacts/model.pt", help="Ruta al modelo entrenado (.pt)")
    ap.add_argument(
        "--forecaster",
        choices=["graphwavenet", "historical", "persistence"],
        default="graphwavenet",
        help="Tipo de modelo de pronóstico a utilizar",
    )
    ap.add_argument("--synthetic", action="store_true", help="Forzar generación de entorno sintético")
    ap.add_argument("--steps", type=int, default=2000, help="Número de pasos de 5 minutos a simular")
    ap.add_argument("--out", default="outputs", help="Directorio para reportes y logs de ejecución")
    ap.add_argument("--noise-std", type=float, default=0.0, help="Desviación de ruido estocástico en velocidades (mph)")
    ap.add_argument("--failure-prob", type=float, default=0.0, help="Probabilidad estocástica de falla de sensor (0 faltante)")
    ap.add_argument("--closed-loop", action="store_true", help="Habilitar retroalimentación del TMS sobre el tráfico")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]

    # 1. Preparar datos y perfil histórico
    if a.synthetic or d.get("synthetic"):
        df, ids, adj = make_synthetic(n_sensors=d.get("synthetic_sensors", 20), days=d.get("synthetic_days", 28))
    else:
        df = load_speed_frame(d["h5_path"])
        ids, adj = load_adjacency(d["adj_path"])

    tr, _, te = chronological_split(df, d["split"])
    profile = HistoricalProfile().fit(tr)

    # 2. Instanciar pronosticador
    if a.forecaster == "graphwavenet":
        fc = TorchForecaster.load(a.model, adj)
    elif a.forecaster == "historical":
        fc = HistoricalAverageForecaster(profile, d["out_len"])
    else:
        fc = PersistenceForecaster(d["out_len"])

    # 3. Preparar directorios y archivos de registro
    os.makedirs(a.out, exist_ok=True)
    tms_log = os.path.join(a.out, "tms_actions.jsonl")
    exec_log = os.path.join(a.out, "execution_log.jsonl")

    # 4. Instanciar Actuador y Agente
    tms = SimulatedTMS(tms_log)
    agent = TrafficFlowAgent(
        forecaster=fc,
        profile=profile,
        sensor_ids=ids,
        cfg=cfg["agent"],
        tms=tms,
        in_len=d["in_len"],
        out_len=d["out_len"],
        execution_log_path=exec_log,
    )

    # 5. Instanciar Entorno vial (TrafficEnvironment)
    env = TrafficEnvironment(
        df=te,
        sensor_ids=ids,
        adj_matrix=adj,
        stochastic_noise_std=a.noise_std,
        sensor_failure_prob=a.failure_prob,
        closed_loop_feedback=a.closed_loop,
    )

    print(f"Iniciando simulación del agente ({a.forecaster.upper()}) sobre {len(ids)} sensores...")
    print(f"Ambiente: {'Sintético' if (a.synthetic or d.get('synthetic')) else 'METR-LA (Real)'} | Pasos: {min(a.steps, len(te))}")

    ts, speeds = env.reset()
    last_action = None

    for _ in range(min(a.steps, len(te))):
        decision = agent.step(ts, speeds)
        action_dict = agent.tms.events[-1] if agent.tms.events else None
        ts, speeds, done, _ = env.step(action=action_dict)
        if done:
            break

    # 6. Generar reportes finales
    rep = agent.report()
    rep_path = os.path.join(a.out, "simulation_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=float)

    m30 = rep["mape_30min"]
    print("\n" + "=" * 60)
    print("RESUMEN DE EJECUCIÓN DEL AGENTE (FASE 2)")
    print("=" * 60)
    print(f"Total pasos simulados:       {rep['total_steps_executed']}")
    print(f"Acciones emitidas a TMS:     {rep['tms_actions']}")
    print(f"Distribución de estados:     {rep['status_counts']}")
    print(f"Recalibraciones adaptativas: {len(rep['recalibrations'])}")
    if m30 is not None:
        meets = "CUMPLE" if rep["meets_target"] else "NO CUMPLE"
        print(f"Error MAPE a 30 min:         {m30:.2%} (Meta SMART < {cfg['agent']['mape_target']:.0%}) -> [{meets}]")
    print(f"\nArchivos generados:")
    print(f" - Log completo de ejecución: {exec_log}")
    print(f" - Acciones registradas TMS:  {tms_log}")
    print(f" - Reporte de simulación:     {rep_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
