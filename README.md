# TrafficFlow-Agent

Agente de predicción de congestión vehicular para un corredor vial piloto. Cada 5 minutos recibe las
velocidades de los sensores, pronostica la velocidad y el nivel de congestión de los **próximos 30 minutos**
(meta: **MAPE < 10 %**), distingue congestión **recurrente** de un **incidente atípico**, actúa sobre un sistema
de gestión de tránsito y se recalibra comparando cada pronóstico con lo realmente observado.

- Datos: [METR-LA](https://github.com/liyaguang/DCRNN) (207 sensores, Los Ángeles, marzo–junio 2012, intervalos de 5 min).
- Diseño: [DL-Traff (arXiv 2108.09091)](https://arxiv.org/abs/2108.09091), de donde se toma el protocolo de evaluación y el modelo base (Graph WaveNet).

## Arquitectura

```
sensores (5 min) ─► ingesta ─► pronosticador (Graph WaveNet) ─► velocidad 5..60 min
                                      │                              │
                                      │                    nivel de congestión (4 niveles)
                                      │                              │
                      perfil histórico (día, franja) ─► recurrente vs. atípico (z-score)
                                                                     │
                                       TMS: planes semafóricos + mensajes a conductores
                                                                     │
        pronóstico vs. observado ─► MAPE móvil a 30 min ─► recalibración si MAPE > 10 %
```

| Módulo | Función |
|---|---|
| `data.py` | Carga METR-LA, división cronológica 70/10/20, escalado y ventanas (12 pasos → 12 pasos) sin fuga de datos |
| `model.py` | Graph WaveNet compacto (convoluciones temporales causales + grafo con adyacencia adaptativa) |
| `forecasters.py` | Pronosticadores intercambiables: Graph WaveNet, promedio histórico, persistencia |
| `profile.py` | Perfil típico por sensor, día de la semana y franja de 5 min |
| `congestion.py` | Velocidad → nivel (fluido / moderado / denso / congestionado) |
| `agent.py` | Ciclo del agente: pronosticar, evaluar, actuar, comparar y recalibrar |
| `actuator.py` | Interfaz con el sistema de gestión de tránsito (simulada en el piloto) |
| `train.py` / `simulate.py` | Entrenamiento con comparación contra líneas base / reproducción del flujo en vivo |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Datos

1. Descarga `metr-la.h5` desde el enlace del README del repositorio DCRNN y guárdalo en `data/metr-la.h5`.
2. Copia `data/sensor_graph/adj_mx.pkl` del mismo repositorio a `data/adj_mx.pkl`.

Los ceros del dataset son datos faltantes: se enmascaran en entrenamiento y en todas las métricas.

## Uso

```bash
python -m trafficflow.train                 # entrena y evalúa (GPU recomendada)
python -m trafficflow.simulate --steps 5000 # reproduce el conjunto de prueba a través del agente
```

Sin datos reales, para probar el flujo completo: añade `--synthetic` a ambos comandos.

Salidas: `artifacts/model.pt`, `artifacts/metrics.json` (MAE/RMSE/MAPE a 15, 30 y 60 min, con líneas base),
`outputs/simulation_report.json` y `outputs/tms_actions.jsonl` (acciones enviadas al TMS).

## Cómo decide el agente

- **Nivel** del corredor: percentil 25 de la velocidad pronosticada (30 min) → fluido ≥ 50 mph, moderado ≥ 35, denso ≥ 20, congestionado < 20.
- **Atípico** si al menos 15 % de los sensores tienen una velocidad pronosticada ≥ 2 desviaciones por debajo de lo típico para ese día y hora; **recurrente** si hay congestión pero dentro del patrón; **normal** en otro caso.
- **Acción**: plan `PICO_RECURRENTE` (+15 % verde) o `INCIDENTE` (+30 % verde) y mensaje a conductores. Solo se envía cuando la situación cambia o cada 30 min para confirmar; al normalizarse se restablece el plan base.
- **Aprendizaje**: cada pronóstico se compara con lo observado; si el MAPE móvil a 30 min (24 h) supera 10 %, se ajusta el modelo con los últimos 7 días (con enfriamiento de 24 h).

Todos los umbrales están en `configs/default.yaml`.

## Limitaciones

- METR-LA es de 2012, solo autopistas y sin etiquetas de incidentes; la detección de atípicos se valida con incidentes inyectados.
- No hay semáforos reales en el dataset: el TMS es simulado (`SimulatedTMS`). Para un sistema real, implementa `TrafficManagementClient`.
- La recalibración es un ajuste fino corto; en producción conviene reentrenar periódicamente con validación.

## Subir a GitHub

```bash
git init && git add . && git commit -m "TrafficFlow-Agent: versión inicial"
gh repo create trafficflow-agent --public --source=. --push   # o crea el repo en github.com y usa git remote add + git push
```
