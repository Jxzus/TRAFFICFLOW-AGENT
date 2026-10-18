"""Servidor web y API REST para despliegue en la nube (Render / Railway / Heroku).

Permite visualizar en vivo el ciclo del agente TrafficFlow-Agent:
Percepción -> Decisión -> Acción -> Aprendizaje y control semafórico.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
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

# Estado global de la aplicación
app_state: dict[str, Any] = {}


def init_agent():
    """Inicializa el agente y el entorno (usando METR-LA si existe, o generador sintético)."""
    cfg_path = "configs/default.yaml"
    cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {"data": {"in_len": 12, "out_len": 12, "split": [0.7, 0.1, 0.2]}, "agent": {"mape_target": 0.10, "eval_horizon_steps": 6, "level_thresholds_mph": [50, 35, 20], "corridor_percentile": 25, "atypical_z": 2.0, "atypical_fraction": 0.15, "drift_window_steps": 288, "drift_min_samples": 20, "recalibration_cooldown_steps": 288, "history_steps": 2016}}
    d = cfg["data"]
    h5_path = d.get("h5_path", "data/metr-la.h5")
    adj_path = d.get("adj_path", "data/adj_mx.pkl")
    model_path = "artifacts/model.pt"

    if os.path.exists(h5_path) and os.path.exists(adj_path):
        try:
            df = load_speed_frame(h5_path)
            ids, adj = load_adjacency(adj_path)
            data_mode = "METR-LA (Real - 207 sensores)"
        except Exception:
            df, ids, adj = make_synthetic(n_sensors=20, days=28)
            data_mode = "Sintético (Emulado - 20 sensores)"
    else:
        df, ids, adj = make_synthetic(n_sensors=20, days=28)
        data_mode = "Sintético (Emulado - 20 sensores)"

    tr, _, te = chronological_split(df, d.get("split", [0.7, 0.1, 0.2]))
    profile = HistoricalProfile().fit(tr)

    if os.path.exists(model_path):
        try:
            fc = TorchForecaster.load(model_path, adj)
        except Exception:
            fc = HistoricalAverageForecaster(profile, d.get("out_len", 12))
    else:
        fc = HistoricalAverageForecaster(profile, d.get("out_len", 12))

    tms = SimulatedTMS()
    agent = TrafficFlowAgent(
        forecaster=fc,
        profile=profile,
        sensor_ids=ids,
        cfg=cfg["agent"],
        tms=tms,
        in_len=d.get("in_len", 12),
        out_len=d.get("out_len", 12),
    )

    env = TrafficEnvironment(
        df=te,
        sensor_ids=ids,
        adj_matrix=adj,
        stochastic_noise_std=1.0,
        sensor_failure_prob=0.02,
        closed_loop_feedback=True,
    )

    ts, speeds = env.reset()
    app_state["agent"] = agent
    app_state["env"] = env
    app_state["cfg"] = cfg
    app_state["ids"] = ids
    app_state["data_mode"] = data_mode
    app_state["current_ts"] = ts
    app_state["current_speeds"] = speeds
    app_state["last_decision"] = None
    app_state["history"] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_agent()
    yield


app = FastAPI(
    title="TrafficFlow-Agent API",
    description="Agente Inteligente de Predicción y Control de Congestión Vehicular",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agent": "TrafficFlow-Agent",
        "version": "1.0.0",
        "mode": app_state.get("data_mode", "Desconocido"),
    }


@app.get("/api/status")
def get_status():
    agent: TrafficFlowAgent = app_state["agent"]
    rep = agent.report()
    return {
        "mode": app_state.get("data_mode"),
        "current_timestamp": str(app_state.get("current_ts")),
        "total_steps": agent.step_i + 1,
        "last_decision": app_state.get("last_decision"),
        "report": rep,
    }


@app.post("/api/step")
@app.get("/api/step")
def run_step():
    agent: TrafficFlowAgent = app_state["agent"]
    env: TrafficEnvironment = app_state["env"]
    ts = app_state["current_ts"]
    speeds = app_state["current_speeds"]

    decision = agent.step(ts, speeds)
    app_state["last_decision"] = {
        "status": decision.status if decision else "Iniciando búfer",
        "level": decision.level if decision else 0,
        "level_name": decision.level_name if decision else "N/A",
        "corridor_speed_mph": round(decision.corridor_speed_mph, 2) if decision else round(float(np.nanmean(speeds)), 2),
        "atypical_fraction": round(decision.atypical_fraction * 100, 1) if decision else 0.0,
        "affected_sensors_count": len(decision.affected_sensors) if decision else 0,
    } if decision else {"status": "Llenando búfer histórico (pasos iniciales)", "level_name": "Inicializando"}

    last_action = agent.tms.events[-1] if agent.tms.events else None
    next_ts, next_speeds, done, _ = env.step(action=last_action)
    if done:
        next_ts, next_speeds = env.reset()

    app_state["current_ts"] = next_ts
    app_state["current_speeds"] = next_speeds

    step_summary = {
        "step": agent.step_i,
        "timestamp": str(ts),
        "observed_mean_speed": round(float(np.nanmean(speeds)), 2),
        "decision": app_state["last_decision"],
        "action_taken": bool(last_action),
        "last_action": last_action,
        "rolling_mape_30min": round(agent.tracker.rolling_mape(agent.h)[0] * 100, 2) if not np.isnan(agent.tracker.rolling_mape(agent.h)[0]) else None,
    }
    app_state["history"].append(step_summary)
    if len(app_state["history"]) > 50:
        app_state["history"].pop(0)

    return step_summary


@app.post("/api/reset")
def reset_simulation():
    init_agent()
    return {"status": "reiniciado con éxito"}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    data_mode = app_state.get("data_mode", "En ejecución")
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficFlow-Agent — Dashboard de Operación</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; }}
        .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }}
        .badge-normal {{ background-color: #10b981; color: white; }}
        .badge-recurrente {{ background-color: #f59e0b; color: white; }}
        .badge-atipica {{ background-color: #ef4444; color: white; }}
        .metric-val {{ font-size: 2rem; font-weight: 700; }}
        pre {{ background: #090d16; color: #38bdf8; padding: 12px; border-radius: 8px; }}
    </style>
</head>
<body class="p-4">
    <div class="container">
        <header class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
            <div>
                <h1 class="h3 mb-0 text-white">🚦 TrafficFlow-Agent</h1>
                <p class="text-secondary small mb-0">Agente de Predicción y Control de Tráfico | UTB Inteligencia Artificial</p>
            </div>
            <div>
                <span class="badge bg-primary px-3 py-2">{data_mode}</span>
                <span class="badge bg-success px-3 py-2">En línea (Render)</span>
            </div>
        </header>

        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <span class="text-secondary small">Velocidad Actual</span>
                    <div id="cur-speed" class="metric-val text-info">-- mph</div>
                    <small id="cur-level" class="text-secondary">Nivel: --</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <span class="text-secondary small">Diagnóstico del Corredor</span>
                    <div id="cur-status" class="metric-val text-white">--</div>
                    <small id="cur-ts" class="text-secondary">Paso: --</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <span class="text-secondary small">MAPE a 30 min (Meta &lt; 10%)</span>
                    <div id="cur-mape" class="metric-val text-success">-- %</div>
                    <small id="cur-target" class="text-secondary">Cumplimiento: --</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <span class="text-secondary small">Plan Semafórico (TMS)</span>
                    <div id="cur-plan" class="metric-val text-warning">BASE</div>
                    <small id="cur-green" class="text-secondary">Extensión verde: 0%</small>
                </div>
            </div>
        </div>

        <div class="d-flex gap-2 mb-4">
            <button onclick="step()" class="btn btn-primary px-4">▶ Avanzar 1 Paso (5 min)</button>
            <button id="btn-auto" onclick="toggleAuto()" class="btn btn-outline-info">▶ Iniciar Simulación Continua</button>
            <button onclick="resetSim()" class="btn btn-outline-danger ms-auto">Reiniciar</button>
        </div>

        <div class="card p-3 mb-4">
            <h5 class="card-title h6 text-white mb-3">Ciclo Agéntico del Último Paso (Percepción → Decisión → Acción)</h5>
            <pre id="log-box" class="mb-0">Presiona "Avanzar 1 Paso" para iniciar el ciclo agéntico...</pre>
        </div>

        <footer class="text-center text-secondary small pt-3">
            Desarrollado por Jesús Andrés Pérez Fuentes y Cristian David Cardeño Gulloso — Universidad Tecnológica de Bolívar
        </footer>
    </div>

    <script>
        let autoInterval = null;
        async function step() {{
            const res = await fetch('/api/step', {{ method: 'POST' }});
            const d = await res.json();
            document.getElementById('cur-speed').innerText = (d.observed_mean_speed || '--') + ' mph';
            document.getElementById('cur-level').innerText = 'Nivel: ' + (d.decision.level_name || 'N/A');
            document.getElementById('cur-status').innerText = (d.decision.status || '--').toUpperCase();
            document.getElementById('cur-ts').innerText = d.timestamp;
            
            const mape = d.rolling_mape_30min;
            if (mape !== null) {{
                document.getElementById('cur-mape').innerText = mape + ' %';
                document.getElementById('cur-target').innerText = mape < 10.0 ? 'Cumple Meta SMART' : 'Deriva detectada';
                document.getElementById('cur-mape').className = mape < 10.0 ? 'metric-val text-success' : 'metric-val text-danger';
            }}
            
            if (d.last_action && d.last_action.plan) {{
                document.getElementById('cur-plan').innerText = d.last_action.plan;
                document.getElementById('cur-green').innerText = 'Extensión verde: +' + (d.last_action.green_extension_pct || 0) + '%';
            }} else {{
                document.getElementById('cur-plan').innerText = 'BASE';
                document.getElementById('cur-green').innerText = 'Extensión verde: 0%';
            }}
            
            document.getElementById('log-box').innerText = JSON.stringify(d, null, 2);
        }}

        function toggleAuto() {{
            const btn = document.getElementById('btn-auto');
            if (autoInterval) {{
                clearInterval(autoInterval);
                autoInterval = null;
                btn.innerText = '▶ Iniciar Simulación Continua';
                btn.className = 'btn btn-outline-info';
            }} else {{
                autoInterval = setInterval(step, 1500);
                btn.innerText = '⏸ Pausar Simulación';
                btn.className = 'btn btn-warning';
            }}
        }}

        async function resetSim() {{
            await fetch('/api/reset', {{ method: 'POST' }});
            alert('Simulación reiniciada');
            location.reload();
        }}
    </script>
</body>
</html>"""
