"""Reproduce el conjunto de prueba como flujo en vivo (una lectura cada 5 min) a través del agente."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from .actuator import SimulatedTMS
from .agent import TrafficFlowAgent
from .data import chronological_split, load_adjacency, load_speed_frame
from .forecasters import HistoricalAverageForecaster, PersistenceForecaster, TorchForecaster
from .profile import HistoricalProfile
from .synthetic import make_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="artifacts/model.pt")
    ap.add_argument("--forecaster", choices=["graphwavenet", "historical", "persistence"], default="graphwavenet")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=2000, help="pasos del conjunto de prueba a reproducir")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]
    if a.synthetic or d.get("synthetic"):
        df, ids, adj = make_synthetic(n_sensors=d.get("synthetic_sensors", 20), days=d.get("synthetic_days", 28))
    else:
        df = load_speed_frame(d["h5_path"])
        ids, adj = load_adjacency(d["adj_path"])
    tr, _, te = chronological_split(df, d["split"])
    profile = HistoricalProfile().fit(tr)
    if a.forecaster == "graphwavenet":
        fc = TorchForecaster.load(a.model, adj)
    elif a.forecaster == "historical":
        fc = HistoricalAverageForecaster(profile, d["out_len"])
    else:
        fc = PersistenceForecaster(d["out_len"])
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "tms_actions.jsonl")
    open(log, "w").close()
    agent = TrafficFlowAgent(fc, profile, ids, cfg["agent"], tms=SimulatedTMS(log), in_len=d["in_len"], out_len=d["out_len"])
    values = te.to_numpy()
    for i in range(min(a.steps, len(te))):
        agent.step(te.index[i], values[i])
    rep = agent.report()
    with open(os.path.join(a.out, "simulation_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    m30 = rep["mape_30min"]
    print(json.dumps({k: rep[k] for k in ("status_counts", "tms_actions", "recalibrations")}, indent=2, default=float))
    print(f"MAPE a 30 min (flujo simulado): {m30:.2%} | meta < {cfg['agent']['mape_target']:.0%}: {'CUMPLE' if rep['meets_target'] else 'NO CUMPLE'}")


if __name__ == "__main__":
    main()
