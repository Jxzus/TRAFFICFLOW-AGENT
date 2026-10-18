"""Servidor web y API REST para despliegue en la nube (Render / Railway / Heroku).

Permite visualizar en vivo el ciclo del agente TrafficFlow-Agent:
Percepción -> Decisión -> Acción -> Aprendizaje y control semafórico.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
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


def to_native(obj: Any) -> Any:
    """Convierte recursivamente tipos de NumPy y Pandas a tipos nativos estándar de Python."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(x) for x in obj]
    if hasattr(obj, "item") and not isinstance(obj, np.ndarray):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        return to_native(obj.to_dict())
    return obj


def safe_json_response(data: Any, status_code: int = 200) -> Response:
    """Serializa cualquier estructura a JSON evitando jsonable_encoder de FastAPI.

    Garantiza compatibilidad total con NumPy y Python 3.12/3.14 en entornos como Render.
    """
    native_data = to_native(data)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    body = json.dumps(native_data, default=_default, ensure_ascii=False)
    return Response(content=body, status_code=status_code, media_type="application/json")


app = FastAPI(
    title="TrafficFlow-Agent API",
    description="Agente Inteligente de Predicción y Control de Congestión Vehicular",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_class=Response)
def health():
    return safe_json_response({
        "status": "healthy",
        "agent": "TrafficFlow-Agent",
        "version": "1.0.0",
        "mode": app_state.get("data_mode", "Desconocido"),
    })


@app.get("/api/status", response_class=Response)
def get_status():
    agent: TrafficFlowAgent = app_state["agent"]
    rep = agent.report()
    data = {
        "mode": app_state.get("data_mode"),
        "current_timestamp": str(app_state.get("current_ts")),
        "total_steps": int(agent.step_i + 1),
        "last_decision": app_state.get("last_decision"),
        "report": rep,
    }
    return safe_json_response(data)


@app.post("/api/step", response_class=Response)
@app.get("/api/step", response_class=Response)
def run_step():
    agent: TrafficFlowAgent = app_state["agent"]
    env: TrafficEnvironment = app_state["env"]
    ts = app_state["current_ts"]
    speeds = app_state["current_speeds"]

    decision = agent.step(ts, speeds)
    app_state["last_decision"] = {
        "status": str(decision.status) if decision else "Iniciando búfer",
        "level": int(decision.level) if decision else 0,
        "level_name": str(decision.level_name) if decision else "N/A",
        "corridor_speed_mph": round(float(decision.corridor_speed_mph), 2) if decision else round(float(np.nanmean(speeds)), 2),
        "atypical_fraction": round(float(decision.atypical_fraction) * 100, 1) if decision else 0.0,
        "affected_sensors_count": int(len(decision.affected_sensors)) if decision else 0,
    } if decision else {"status": "Llenando búfer histórico (pasos iniciales)", "level_name": "Inicializando"}

    last_action = agent.tms.events[-1] if agent.tms.events else None
    next_ts, next_speeds, done, _ = env.step(action=last_action)
    if done:
        next_ts, next_speeds = env.reset()

    app_state["current_ts"] = next_ts
    app_state["current_speeds"] = next_speeds

    mape_tuple = agent.tracker.rolling_mape(agent.h)
    mape_val = float(mape_tuple[0]) if (mape_tuple and not np.isnan(mape_tuple[0])) else None

    step_summary = {
        "step": int(agent.step_i),
        "timestamp": str(ts),
        "observed_mean_speed": round(float(np.nanmean(speeds)), 2),
        "decision": app_state["last_decision"],
        "action_taken": bool(last_action),
        "last_action": last_action,
        "rolling_mape_30min": round(mape_val * 100, 2) if mape_val is not None else None,
    }
    app_state["history"].append(step_summary)
    if len(app_state["history"]) > 50:
        app_state["history"].pop(0)

    return safe_json_response(step_summary)


@app.post("/api/reset", response_class=Response)
def reset_simulation():
    init_agent()
    return safe_json_response({"status": "reiniciado con éxito"})


@app.get("/", response_class=HTMLResponse)
def dashboard():
    data_mode = app_state.get("data_mode", "En ejecución")
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficFlow-Agent — Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {{
            --bg-base:#0f172a; --bg-card:#1e293b; --bg-card2:#162032;
            --border:#334155; --muted:#94a3b8;
        }}
        body {{ background:var(--bg-base); color:#f8fafc; font-family:system-ui,sans-serif; }}
        .card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; }}
        .metric-val {{ font-size:2rem; font-weight:700; }}
        pre {{ background:#090d16; color:#38bdf8; padding:12px; border-radius:8px;
               font-size:.78rem; max-height:280px; overflow-y:auto; }}
        /* Help button */
        .help-btn {{
            display:inline-flex; align-items:center; justify-content:center;
            width:18px; height:18px; border-radius:50%;
            background:#334155; color:var(--muted); font-size:11px; font-weight:700;
            border:none; cursor:pointer; margin-left:4px; line-height:1; transition:background .2s;
        }}
        .help-btn:hover {{ background:#475569; color:#f8fafc; }}
        /* Status badges */
        .badge-normal    {{ background:#10b981; color:#fff; }}
        .badge-recurrente{{ background:#f59e0b; color:#fff; }}
        .badge-atipica   {{ background:#ef4444; color:#fff; }}
        /* Cycle diagram */
        .cycle-step {{ display:flex; flex-direction:column; align-items:center; flex:1; min-width:0; }}
        .cycle-icon {{
            width:48px; height:48px; border-radius:50%;
            display:flex; align-items:center; justify-content:center;
            font-size:1.3rem; margin-bottom:4px;
            border:2px solid var(--border); background:var(--bg-card2);
            transition:border-color .4s, box-shadow .4s;
        }}
        .cycle-icon.active {{ border-color:#38bdf8; box-shadow:0 0 12px #38bdf855; }}
        .cycle-arrow {{ color:var(--muted); font-size:1.3rem; padding:0 4px; margin-top:14px; }}
        .cycle-label {{ font-size:.7rem; color:var(--muted); text-align:center; }}
        /* Section labels */
        .sec-label {{
            font-size:.65rem; font-weight:700; letter-spacing:.08em;
            color:var(--muted); text-transform:uppercase; margin-bottom:8px;
        }}
        /* Accordion */
        .accordion-button {{ background:var(--bg-card2)!important; color:#f8fafc!important; }}
        .accordion-button:not(.collapsed) {{ background:#1a2e4a!important; color:#38bdf8!important; }}
        .accordion-button::after {{ filter:invert(1); }}
        .accordion-body {{ background:var(--bg-card2); color:#cbd5e1; font-size:.875rem; }}
        .tooltip-inner {{ max-width:260px; text-align:left; }}
        @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.4}} }}
        .pulsing {{ animation:pulse 1.2s ease-in-out infinite; }}
    </style>
</head>
<body class="p-3 p-md-4">
<div class="container-xl">

<!-- ═══ HEADER ═══ -->
<header class="d-flex flex-wrap justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary gap-2">
    <div>
        <h1 class="h3 mb-0 text-white">🚦 TrafficFlow-Agent</h1>
        <p class="text-secondary small mb-0">Agente Inteligente de Predicción y Control de Congestión Vehicular · UTB IA 2026-2</p>
    </div>
    <div class="d-flex align-items-center gap-2 flex-wrap">
        <span class="badge bg-primary px-3 py-2">{data_mode}</span>
        <span class="badge bg-success px-3 py-2">🟢 En línea (Render)</span>
        <span class="badge bg-secondary px-3 py-2" id="step-badge">Pasos: 0</span>
        <button class="btn btn-outline-secondary btn-sm" data-bs-toggle="modal" data-bs-target="#helpModal">❓ Guía de uso</button>
    </div>
</header>

<!-- ═══ CICLO AGÉNTICO ═══ -->
<div class="card p-3 mb-4">
    <div class="d-flex justify-content-between align-items-center mb-2">
        <span class="sec-label">Ciclo Agéntico — Paso actual</span>
        <button class="help-btn" data-bs-toggle="tooltip" title="El agente ejecuta 4 fases en cada paso de 5 min: Percepción (lee sensores), Decisión (predice 30 min), Acción (ajusta semáforos) y Aprendizaje (evalúa MAPE).">?</button>
    </div>
    <div class="d-flex align-items-start justify-content-center py-2">
        <div class="cycle-step">
            <div class="cycle-icon" id="ic-perceive">👁️</div>
            <div class="cycle-label">1. Percepción<br><small>Lee sensores</small></div>
        </div>
        <div class="cycle-arrow">→</div>
        <div class="cycle-step">
            <div class="cycle-icon" id="ic-decide">🧠</div>
            <div class="cycle-label">2. Decisión<br><small>Predice 30 min</small></div>
        </div>
        <div class="cycle-arrow">→</div>
        <div class="cycle-step">
            <div class="cycle-icon" id="ic-act">🚦</div>
            <div class="cycle-label">3. Acción<br><small>Ajusta TMS</small></div>
        </div>
        <div class="cycle-arrow">→</div>
        <div class="cycle-step">
            <div class="cycle-icon" id="ic-learn">📈</div>
            <div class="cycle-label">4. Aprendizaje<br><small>Evalúa MAPE</small></div>
        </div>
    </div>
    <div id="cycle-msg" class="text-center text-white small mt-1">Presiona «Avanzar» para iniciar el ciclo agéntico.</div>
</div>

<!-- ═══ MÉTRICAS ═══ -->
<div class="row g-3 mb-4">
    <div class="col-sm-6 col-xl-3">
        <div class="card p-3 text-center h-100">
            <div class="d-flex justify-content-center align-items-center gap-1 mb-1">
                <span class="text-secondary small">Velocidad Promedio</span>
                <button class="help-btn" data-bs-toggle="tooltip" title="Promedio de velocidad (mph) en todos los sensores activos. &gt;50 mph: fluido · 35–50: lento · &lt;35: congestionado.">?</button>
            </div>
            <div id="cur-speed" class="metric-val text-info">-- mph</div>
            <small id="cur-level" class="text-secondary">Nivel: --</small>
        </div>
    </div>
    <div class="col-sm-6 col-xl-3">
        <div class="card p-3 text-center h-100">
            <div class="d-flex justify-content-center align-items-center gap-1 mb-1">
                <span class="text-secondary small">Diagnóstico del Corredor</span>
                <button class="help-btn" data-bs-toggle="tooltip" title="NORMAL: flujo libre · RECURRENTE: hora pico (patrón conocido) · ATÍPICA: posible incidente (z-score &gt; 2σ fuera del patrón histórico).">?</button>
            </div>
            <div id="cur-status" class="metric-val text-white">--</div>
            <small id="cur-ts" class="text-secondary">Timestamp: --</small>
        </div>
    </div>
    <div class="col-sm-6 col-xl-3">
        <div class="card p-3 text-center h-100">
            <div class="d-flex justify-content-center align-items-center gap-1 mb-1">
                <span class="text-secondary small">MAPE a 30 min</span>
                <button class="help-btn" data-bs-toggle="tooltip" title="Error porcentual absoluto medio del pronóstico a 30 min. Meta SMART: MAPE &lt; 10%. Ventana deslizante de 288 pasos (~24 h). Verde = OK · Rojo = deriva detectada.">?</button>
            </div>
            <div id="cur-mape" class="metric-val text-success">-- %</div>
            <small id="cur-target" class="text-secondary">Cumplimiento: --</small>
        </div>
    </div>
    <div class="col-sm-6 col-xl-3">
        <div class="card p-3 text-center h-100">
            <div class="d-flex justify-content-center align-items-center gap-1 mb-1">
                <span class="text-secondary small">Plan Semafórico (TMS)</span>
                <button class="help-btn" data-bs-toggle="tooltip" title="BASE: ciclo normal · PICO_RECURRENTE: verde +15% (hora pico) · INCIDENTE: verde +30% (congestión atípica). Solo cambia cuando cambia el diagnóstico del corredor.">?</button>
            </div>
            <div id="cur-plan" class="metric-val text-warning">BASE</div>
            <small id="cur-green" class="text-secondary">Extensión verde: 0%</small>
        </div>
    </div>
</div>

<!-- ═══ CONTROLES ═══ -->
<div class="card p-3 mb-4">
    <div class="d-flex justify-content-between align-items-center mb-2">
        <span class="sec-label">Controles de simulación</span>
        <button class="help-btn" data-bs-toggle="tooltip" title="Cada paso = 5 min reales de datos. La simulación continua avanza cada 1.5 s. Usa +10 Pasos para llegar rápido a estados de congestión.">?</button>
    </div>
    <div class="d-flex gap-2 flex-wrap">
        <button onclick="stepOnce()" class="btn btn-primary px-4">▶ Avanzar 1 Paso (5 min)</button>
        <button id="btn-auto" onclick="toggleAuto()" class="btn btn-outline-info">⏵ Simulación Continua</button>
        <button onclick="step10()" class="btn btn-outline-secondary">⏭ +10 Pasos</button>
        <button onclick="resetSim()" class="btn btn-outline-danger ms-auto">🔄 Reiniciar</button>
    </div>
    <div id="status-bar" class="mt-2 small text-muted d-none"><span class="pulsing">⬤</span> Ejecutando paso...</div>
</div>

<!-- ═══ LOG + SENSORES ═══ -->
<div class="row g-3 mb-4">
    <div class="col-lg-7">
        <div class="card p-3 h-100">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span class="sec-label">Respuesta JSON del último paso</span>
                <button class="help-btn" data-bs-toggle="tooltip" title="Respuesta de POST /api/step: step (intervalo), timestamp, observed_mean_speed (velocidad medida), decision (diagnóstico), last_action (plan TMS), rolling_mape_30min (error del pronóstico).">?</button>
            </div>
            <pre id="log-box">Presiona «Avanzar 1 Paso» para ver el ciclo agéntico en acción...</pre>
        </div>
    </div>
    <div class="col-lg-5">
        <div class="card p-3 h-100">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span class="sec-label">Sensores afectados</span>
                <button class="help-btn" data-bs-toggle="tooltip" title="IDs de sensores METR-LA con velocidad bajo umbral (nivel ≥ 2) o z-score atípico &gt; 2σ. En estado NORMAL la lista está vacía.">?</button>
            </div>
            <div id="sensors-box" class="d-flex flex-wrap gap-1" style="min-height:60px">
                <span class="text-muted small">Sin datos aún</span>
            </div>
            <hr class="border-secondary my-2">
            <div class="sec-label mb-1">Últimas acciones TMS
                <button class="help-btn" data-bs-toggle="tooltip" title="Registro cronológico de las órdenes enviadas al Sistema de Gestión de Tránsito (TMS). Solo se registran cuando action_taken=true.">?</button>
            </div>
            <div id="actions-log" class="small" style="max-height:130px;overflow-y:auto;color:#94a3b8">
                <em>Sin acciones emitidas aún.</em>
            </div>
        </div>
    </div>
</div>

<!-- ═══ INSTRUCCIONES ═══ -->
<div class="accordion mb-4" id="acc">
    <div class="accordion-item border-0" style="border-radius:12px;overflow:hidden">
        <h2 class="accordion-header">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#colGuide">
                📖 Instrucciones de uso y casos de prueba sugeridos
            </button>
        </h2>
        <div id="colGuide" class="accordion-collapse collapse" data-bs-parent="#acc">
            <div class="accordion-body">
                <div class="row g-4">
                    <div class="col-md-6">
                        <h6 class="text-info">🧭 ¿Cómo usar el dashboard?</h6>
                        <ol class="ps-3 mb-0">
                            <li class="mb-1"><strong>Avanzar 1 Paso</strong> — ejecuta un intervalo de 5 min de datos reales METR-LA y muestra el resultado del ciclo agéntico completo.</li>
                            <li class="mb-1"><strong>Simulación Continua</strong> — avanza pasos automáticamente cada 1.5 s; ideal para observar transiciones de estado en tiempo real.</li>
                            <li class="mb-1"><strong>+10 Pasos</strong> — avance rápido hacia estados de congestión sin esperar paso a paso.</li>
                            <li class="mb-1"><strong>Reiniciar</strong> — vuelve al paso 0 con el agente limpio para repetir el experimento.</li>
                            <li class="mb-1">Cada tarjeta y sección tiene un botón <strong>?</strong> con explicación detallada de la métrica.</li>
                            <li class="mb-1">Pulsa <strong>❓ Guía de uso</strong> (arriba) para la guía completa con tabla de casos de prueba.</li>
                        </ol>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-warning">🧪 Casos de prueba sugeridos</h6>
                        <table class="table table-sm table-dark table-borderless mb-0" style="font-size:.8rem">
                            <thead><tr><th>Caso</th><th>Acción</th><th>Resultado esperado</th></tr></thead>
                            <tbody>
                                <tr><td><span class="badge badge-normal">Normal</span></td><td>1 paso</td><td>NORMAL, plan BASE, MAPE acumulándose</td></tr>
                                <tr><td><span class="badge badge-recurrente">Hora pico</span></td><td>+10 Pasos</td><td>RECURRENTE, PICO_RECURRENTE, +15% verde</td></tr>
                                <tr><td><span class="badge badge-atipica">Incidente</span></td><td>Simul. continua</td><td>ATÍPICA, INCIDENTE, +30% verde, sensores en rojo</td></tr>
                                <tr><td>📉 MAPE visible</td><td>30+ pasos</td><td>MAPE &lt; 10% (meta SMART cumplida)</td></tr>
                                <tr><td>🔄 Reset</td><td>Reiniciar → paso 1</td><td>Estado inicial limpio y reproducible</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <hr class="border-secondary">
                <div class="row g-3">
                    <div class="col-md-6">
                        <h6 class="text-success">🔌 Endpoints de la API pública</h6>
                        <ul class="ps-3 mb-0 small">
                            <li><code>GET  /health</code> — Estado del servidor</li>
                            <li><code>GET  /api/status</code> — Estado completo del agente y métricas acumuladas</li>
                            <li><code>POST /api/step</code> — Ejecuta 1 ciclo agéntico completo (5 min)</li>
                            <li><code>GET  /api/step</code> — Mismo efecto (compatible con navegador)</li>
                            <li><code>POST /api/reset</code> — Reinicia el agente y el entorno</li>
                        </ul>
                        <code class="d-block mt-2 p-2" style="background:#090d16;border-radius:6px;font-size:.75rem;color:#38bdf8">curl -X POST https://&lt;url-render&gt;/api/step</code>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-primary">🗂️ Arquitectura del agente</h6>
                        <ul class="ps-3 mb-0 small">
                            <li><strong>Tipo:</strong> Agente basado en modelo (AIMA) + aprendizaje continuo</li>
                            <li><strong>Datos:</strong> {data_mode}</li>
                            <li><strong>Pronóstico:</strong> Historical Average / Graph WaveNet</li>
                            <li><strong>Horizonte:</strong> 30 min (6 pasos × 5 min)</li>
                            <li><strong>Meta SMART:</strong> MAPE &lt; 10% en ventana de 24 h</li>
                            <li><strong>Actuador:</strong> SimulatedTMS (semáforos digitales)</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- ═══ FOOTER ═══ -->
<footer class="text-center text-secondary small pt-2 pb-4">
    <strong>Jesús Andrés Pérez Fuentes</strong> (T00078491) · <strong>Cristian David Cardeño Gulloso</strong> (T00078446) —
    Universidad Tecnológica de Bolívar · Prof. Edwin A. Puertas Del Castillo · 2026-2
</footer>
</div>

<!-- ═══ MODAL GUÍA COMPLETA ═══ -->
<div class="modal fade" id="helpModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-scrollable">
        <div class="modal-content" style="background:var(--bg-card);color:#f8fafc;border:1px solid var(--border)">
            <div class="modal-header border-secondary">
                <h5 class="modal-title">❓ Guía completa de uso — TrafficFlow-Agent</h5>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
            </div>
            <div class="modal-body" style="font-size:.88rem">
                <h6 class="text-info">¿Qué es TrafficFlow-Agent?</h6>
                <p>Agente inteligente que monitorea velocidades de tráfico en la red vial de Los Ángeles (datos METR-LA, 207 sensores) y decide en tiempo real si ajustar los tiempos de los semáforos para mitigar la congestión.</p>

                <h6 class="text-warning mt-3">Flujo de operación (4 fases por paso de 5 min)</h6>
                <ol>
                    <li><strong>Percepción:</strong> Lee velocidades de los 207 sensores, detecta datos faltantes y registra el estado de la red.</li>
                    <li><strong>Decisión:</strong> Pronostica el estado del corredor a 30 min y calcula el z-score respecto al perfil histórico de 2016 pasos.</li>
                    <li><strong>Acción:</strong> Si hay congestión RECURRENTE o ATÍPICA, envía un plan semafórico al TMS simulado.</li>
                    <li><strong>Aprendizaje:</strong> Evalúa el MAPE rolling. Si supera el 10%, activa recalibración adaptativa del modelo.</li>
                </ol>

                <h6 class="text-success mt-3">Estados del corredor</h6>
                <ul>
                    <li><span class="badge badge-normal">NORMAL</span> — Tráfico fluido, sin intervención. Plan: BASE.</li>
                    <li><span class="badge badge-recurrente">RECURRENTE</span> — Hora pico esperada según patrón histórico. Plan: PICO_RECURRENTE (+15% tiempo verde).</li>
                    <li><span class="badge badge-atipica">ATÍPICA</span> — Congestión inusual (accidente, evento). Plan: INCIDENTE (+30% tiempo verde). Sensores afectados resaltados en rojo.</li>
                </ul>

                <h6 class="text-primary mt-3">Tabla de casos de prueba</h6>
                <table class="table table-sm table-dark">
                    <thead><tr><th>Nº</th><th>Escenario</th><th>Cómo llegar</th><th>Resultado esperado</th></tr></thead>
                    <tbody>
                        <tr><td>1</td><td>Arranque del agente</td><td>Pasos 1–11</td><td>Búfer histórico llenándose; diagnóstico vacío; MAPE N/A</td></tr>
                        <tr><td>2</td><td>Primera decisión</td><td>Paso 12</td><td>Primer diagnóstico visible (NORMAL, RECURRENTE o ATÍPICA)</td></tr>
                        <tr><td>3</td><td>Congestión hora pico</td><td>+10 Pasos (7–9 AM)</td><td>RECURRENTE, plan PICO_RECURRENTE activo</td></tr>
                        <tr><td>4</td><td>Incidente atípico</td><td>Simulación continua</td><td>ATÍPICA, plan INCIDENTE, sensores afectados en rojo</td></tr>
                        <tr><td>5</td><td>Meta SMART disponible</td><td>30+ pasos</td><td>MAPE aparece y debe ser &lt; 10%</td></tr>
                        <tr><td>6</td><td>Reproducibilidad</td><td>Reiniciar → Paso 1</td><td>Agente vuelve al estado inicial limpiamente</td></tr>
                    </tbody>
                </table>

                <h6 class="text-secondary mt-3">API pública (sin autenticación)</h6>
                <table class="table table-sm table-dark">
                    <thead><tr><th>Método</th><th>Endpoint</th><th>Descripción</th></tr></thead>
                    <tbody>
                        <tr><td><span class="badge bg-success">GET</span></td><td><code>/health</code></td><td>Estado del servidor</td></tr>
                        <tr><td><span class="badge bg-success">GET</span></td><td><code>/api/status</code></td><td>Estado completo del agente y métricas acumuladas</td></tr>
                        <tr><td><span class="badge bg-primary">POST</span></td><td><code>/api/step</code></td><td>Ejecuta 1 ciclo agéntico (5 min) y retorna el resumen JSON</td></tr>
                        <tr><td><span class="badge bg-success">GET</span></td><td><code>/api/step</code></td><td>Mismo efecto (útil desde el navegador)</td></tr>
                        <tr><td><span class="badge bg-primary">POST</span></td><td><code>/api/reset</code></td><td>Reinicia el agente y el entorno desde cero</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="modal-footer border-secondary">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    // Init Bootstrap tooltips
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el =>
        new bootstrap.Tooltip(el, {{placement:'top', html:true}}));

    let autoInterval = null;
    let stepCount = 0;
    let actionLog = [];

    function animateCycle() {{
        const icons = ['ic-perceive','ic-decide','ic-act','ic-learn'];
        const msgs  = [
            '👁️ Percepción: leyendo velocidades de los sensores...',
            '🧠 Decisión: proyectando velocidad a 30 minutos...',
            '🚦 Acción: evaluando y enviando plan semafórico TMS...',
            '📈 Aprendizaje: calculando MAPE y verificando deriva...',
        ];
        icons.forEach(id => document.getElementById(id).classList.remove('active'));
        document.getElementById('cycle-msg').textContent = msgs[0];
        let i = 0;
        const iv = setInterval(() => {{
            document.getElementById(icons[i]).classList.remove('active');
            i++;
            if (i >= icons.length) {{ clearInterval(iv); return; }}
            document.getElementById(icons[i]).classList.add('active');
            document.getElementById('cycle-msg').textContent = msgs[i];
        }}, 270);
    }}

    async function doStep() {{
        document.getElementById('status-bar').classList.remove('d-none');
        animateCycle();

        const res = await fetch('/api/step', {{method:'POST'}});
        const d   = await res.json();
        document.getElementById('status-bar').classList.add('d-none');

        stepCount++;
        document.getElementById('step-badge').textContent = 'Pasos: ' + stepCount;

        // Velocidad
        document.getElementById('cur-speed').innerText = (d.observed_mean_speed ?? '--') + ' mph';
        document.getElementById('cur-level').innerText = 'Nivel: ' + (d.decision?.level_name ?? 'N/A');

        // Estado + color
        const st   = (d.decision?.status ?? '').toUpperCase();
        const stEl = document.getElementById('cur-status');
        stEl.innerText = st || '--';
        const colors = {{'NORMAL':'text-success','RECURRENTE':'text-warning','ATIPICA':'text-danger','ATÍPICA':'text-danger'}};
        stEl.className = 'metric-val ' + (colors[st] ?? 'text-white');
        document.getElementById('cur-ts').innerText = d.timestamp ?? '--';

        // MAPE
        const mape = d.rolling_mape_30min;
        if (mape !== null && mape !== undefined) {{
            const ok = mape < 10.0;
            document.getElementById('cur-mape').innerText   = mape + ' %';
            document.getElementById('cur-mape').className   = 'metric-val ' + (ok ? 'text-success' : 'text-danger');
            document.getElementById('cur-target').innerText = ok ? '✅ Cumple Meta SMART' : '⚠️ Deriva detectada';
        }}

        // Plan TMS
        if (d.last_action && d.last_action.plan) {{
            document.getElementById('cur-plan').innerText  = d.last_action.plan;
            document.getElementById('cur-green').innerText = 'Extensión verde: +' + (d.last_action.green_extension_pct ?? 0) + '%';
            if (d.action_taken) {{
                const ts = (d.timestamp ?? '').slice(11,19);
                actionLog.unshift('[' + ts + '] ' + d.last_action.plan + ' — ' + (d.last_action.message ?? ''));
                if (actionLog.length > 8) actionLog.pop();
                document.getElementById('actions-log').innerHTML =
                    actionLog.map(a => '<div class="text-truncate">• ' + a + '</div>').join('');
            }}
        }} else {{
            document.getElementById('cur-plan').innerText  = 'BASE';
            document.getElementById('cur-green').innerText = 'Extensión verde: 0%';
        }}

        // Sensores afectados
        const sBox = document.getElementById('sensors-box');
        if (d.decision?.affected_sensors_count > 0 && d.last_action?.sensors?.length) {{
            sBox.innerHTML = d.last_action.sensors
                .map(s => '<span class="badge badge-atipica me-1 mb-1">' + s + '</span>').join('');
        }} else {{
            sBox.innerHTML = '<span class="text-muted small">✅ Ningún sensor en alerta</span>';
        }}

        // JSON raw
        document.getElementById('log-box').innerText = JSON.stringify(d, null, 2);

        // Limpiar ciclo al terminar
        setTimeout(() => {{
            ['ic-perceive','ic-decide','ic-act','ic-learn'].forEach(id =>
                document.getElementById(id).classList.remove('active'));
            document.getElementById('cycle-msg').textContent =
                '✅ Paso ' + stepCount + ' completado · Estado: ' + (st || 'N/A');
        }}, 1150);
    }}

    function stepOnce() {{ doStep(); }}

    async function step10() {{
        for (let i = 0; i < 10; i++) {{ await doStep(); }}
    }}

    function toggleAuto() {{
        const btn = document.getElementById('btn-auto');
        if (autoInterval) {{
            clearInterval(autoInterval); autoInterval = null;
            btn.innerHTML = '⏵ Simulación Continua';
            btn.className = 'btn btn-outline-info';
        }} else {{
            autoInterval = setInterval(doStep, 1500);
            btn.innerHTML = '⏸ Pausar';
            btn.className = 'btn btn-warning';
        }}
    }}

    async function resetSim() {{
        if (!confirm('¿Reiniciar la simulación desde el paso 0?')) return;
        await fetch('/api/reset', {{method:'POST'}});
        stepCount = 0; actionLog = [];
        document.getElementById('step-badge').textContent  = 'Pasos: 0';
        document.getElementById('log-box').innerText       = 'Simulación reiniciada. Presiona «Avanzar 1 Paso» para continuar.';
        document.getElementById('sensors-box').innerHTML   = '<span class="text-muted small">Sin datos aún</span>';
        document.getElementById('actions-log').innerHTML   = '<em>Sin acciones emitidas aún.</em>';
        document.getElementById('cycle-msg').textContent   = 'Presiona «Avanzar» para iniciar el ciclo agéntico.';
        ['ic-perceive','ic-decide','ic-act','ic-learn'].forEach(id =>
            document.getElementById(id).classList.remove('active'));
        if (autoInterval) {{
            clearInterval(autoInterval); autoInterval = null;
            document.getElementById('btn-auto').innerHTML = '⏵ Simulación Continua';
            document.getElementById('btn-auto').className = 'btn btn-outline-info';
        }}
    }}
</script>
</body>
</html>"""
