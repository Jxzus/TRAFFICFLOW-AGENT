# TrafficFlow-Agent: Agente de Predicción y Control de Congestión Vehicular

Agente inteligente basado en modelo con capacidad de aprendizaje (*Model-based Learning Agent*, AIMA Russell & Norvig) para un corredor vial piloto instrumentado con sensores de velocidad. Cada 5 minutos percibe las lecturas de velocidad de los sensores, pronostica el nivel de congestión de los **próximos 30 minutos** (meta SMART: **MAPE < 10 %**), distingue congestión **recurrente** de un **incidente atípico**, actúa sobre el sistema de gestión de tránsito (TMS) y se recalibra adaptativamente comparando cada pronóstico con lo realmente observado.

- **Datos:** [METR-LA](https://github.com/liyaguang/DCRNN) (207 sensores, Los Ángeles, marzo–junio 2012, intervalos de 5 min).
- **Diseño:** Marco AIMA y [DL-Traff (arXiv 2108.09091)](https://arxiv.org/abs/2108.09091), de donde se toma el protocolo de evaluación y la arquitectura base (Graph WaveNet).
- **Entregables por Fase:**
  - **Fase 1 (Selección Justificada):** [FASE_1_SELECCION_JUSTIFICADA_AGENTE.md](FASE_1_SELECCION_JUSTIFICADA_AGENTE.md) | [Word .docx](FASE_1_SELECCION_JUSTIFICADA_AGENTE.docx)
  - **Fase 2 (Desarrollo del Agente):** [FASE_2_DESARROLLO_DEL_AGENTE.md](FASE_2_DESARROLLO_DEL_AGENTE.md) | [Word .docx](FASE_2_DESARROLLO_DEL_AGENTE.docx)
  - **Fase 3 (Pruebas de Desempeño):** [FASE_3_PRUEBAS_MEDIDAS_DESEMPENO.md](FASE_3_PRUEBAS_MEDIDAS_DESEMPENO.md) | [Word .docx](FASE_3_PRUEBAS_MEDIDAS_DESEMPENO.docx)

---

## Arquitectura del Ciclo Agéntico (AIMA)

```
sensores (5 min) ─► [PERCEPCIÓN] ─► ingesta, limpieza de faltantes y memoria de corto/largo plazo
                          │
                          ▼
                    [DECISIÓN]   ─► pronosticador (Graph WaveNet) ─► velocidad 5..60 min
                          │               │
                          │         perfil histórico (día, franja) ─► z-score (recurrente vs. atípico)
                          │               │
                          ▼               ▼
                     [ACCIÓN]    ─► TMS: planes semafóricos (+15 % / +30 % verde) + mensajes a conductores
                          │
                          ▼
                   [APRENDIZAJE] ─► crítica: pronóstico vs. observado a 30 min (MAPE móvil en 24 h)
                                    recalibración adaptativa (fine-tuning) si MAPE > 10 %
```

| Módulo | Función en la Arquitectura |
|---|---|
| `agent.py` | Implementación del agente: ciclo explícito `perceive()` $\to$ `decide()` $\to$ `act()` $\to$ `learn()`, y registro de logs auditables. |
| `environment.py` | Entorno vial (`TrafficEnvironment`) real o simulado (estocástico, parcialmente observable, secuencial y dinámico). |
| `data.py` | Carga de METR-LA, partición cronológica 70/10/20, escalado y ventanas deslizantes sin fuga de datos. |
| `model.py` | Graph WaveNet compacto (convoluciones temporales causales dilatadas + difusión adaptativa sobre grafos). |
| `forecasters.py` | Modelos de pronóstico intercambiables: Graph WaveNet, promedio histórico y persistencia. |
| `profile.py` | Memoria de largo plazo: perfil típico por sensor, día de la semana y franja de 5 min. |
| `congestion.py` | Evaluación analítica: velocidad $\to$ nivel (fluido / moderado / denso / congestionado). |
| `actuator.py` | Interfaz con el sistema de gestión de tránsito (TMS) y mensajería a conductores. |
| `synthetic.py` | Generador sintético de flujos viales con propagación espacial de congestión e incidentes inyectados. |
| `simulate.py` | Simulador de flujo en vivo paso a paso a través del agente con generación de logs para Fase 3. |
| `train.py` | Entrenamiento supervisado y evaluación comparativa contra líneas base. |

---

## Instalación y Configuración

### 1. Clonar el repositorio y crear entorno virtual
```bash
git clone https://github.com/Jxzus/TRAFFICFLOW-AGENT.git
cd TRAFFICFLOW-AGENT-1

python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux / macOS:
# source .venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Verificar la instalación con las pruebas unitarias
```bash
pytest -v
```
*(8 pruebas pasan exitosamente: causalidad temporal, niveles de congestión, detección de incidentes, ciclo explícito, entorno estocástico y recalibración por deriva).*

---

## Datos

El repositorio ya incluye soporte directo para:
1. **Datos reales:** Si se dispone de `metr-la.h5` y `adj_mx.pkl`, deben ubicarse en `data/metr-la.h5` y `data/adj_mx.pkl`.
2. **Entorno sintético:** Si no se descargan los archivos reales, se puede ejecutar cualquier comando añadiendo el parámetro `--synthetic`.

---

## Uso y Ejecución del Agente

### Reproducir el flujo en vivo a través del agente (Simulación)

**Ejecución estándar sobre datos reales:**
```bash
python -m trafficflow.simulate --steps 500 --out outputs
```

**Ejecución con entorno sintético estocástico y lazo cerrado (*Closed-Loop*):**
```bash
python -m trafficflow.simulate --synthetic --noise-std 2.0 --failure-prob 0.05 --closed-loop --steps 500 --out outputs
```

### Entrenar o evaluar el modelo
```bash
python -m trafficflow.train                 # con datos METR-LA
python -m trafficflow.train --synthetic     # con datos sintéticos
```

---

## Registro (Log) de Ejecución y Salidas

Cada ejecución genera automáticamente en el directorio `outputs/`:
1. `outputs/execution_log.jsonl`: **Registro completo de ejecución** (percepciones, decisiones, acciones y retroalimentación de aprendizaje con marcas de tiempo ISO), base para los experimentos de la **Fase 3**.
2. `outputs/tms_actions.jsonl`: Bitácora específica de planes semafóricos y alertas enviados a los actuadores.
3. `outputs/simulation_report.json`: Reporte cuantitativo con el MAPE a 15, 30 y 60 min, desglose de estados y cumplimiento de la meta SMART.

---

## Cómo Decide el Agente

- **Nivel de congestión:** Percentil 25 de la velocidad pronosticada a 30 min $\to$ fluido $\ge 50$ mph, moderado $\ge 35$, denso $\ge 20$, congestionado $< 20$.
- **Atípico vs. Recurrente:** Se calculan las desviaciones típicas ($z$-score) de la predicción respecto a la media histórica del mismo día y hora. Si al menos el $15\%$ de los sensores presentan $z < -2.0$, se clasifica como `atipica` (posible incidente); si hay congestión pero sigue el patrón histórico, se clasifica como `recurrente`.
- **Acción:** Plan `PICO_RECURRENTE` (+15 % verde) o `INCIDENTE` (+30 % verde) y mensajes informativos a paneles viales. Solo se envía ante cambios de estado o cada 30 min para confirmación; al normalizarse se restablece el plan base.
- **Aprendizaje:** El agente rastrea el error relativo a 30 min. Si el MAPE móvil a 30 min en 24 h supera el $10\%$, dispara recalibración (*fine-tuning*) con los últimos 7 días retenidos en memoria.

---

## Licencia y Créditos

Desarrollado para el curso de **Inteligencia Artificial** — Universidad Tecnológica de Bolívar (UTB).  
Autor: Jesús Andrés Pérez Fuentes.
