# UNIVERSIDAD TECNOLÓGICA DE BOLÍVAR
### FACULTAD DE INGENIERÍA — PROGRAMA DE INGENIERÍA DE SISTEMAS
### CURSO: INTELIGENCIA ARTIFICIAL

---

# DISEÑO AJUSTADO DE UN AGENTE INTELIGENTE
## TRAFFICFLOW-AGENT — Agente de Predicción y Control de Congestión Vehicular

- **Presentado por:**
  - Jesús Andrés Pérez Fuentes — **T00078491**
  - Cristian David Cardeño Gulloso — **T00078446**
- **Docente:** Edwin Alexander Puertas Del Castillo
- **Fecha:** Octubre de 2026
- **Cartagena de Indias, Colombia**
- **Repositorio:** [https://github.com/Jxzus/TRAFFICFLOW-AGENT](https://github.com/Jxzus/TRAFFICFLOW-AGENT)

---

## TABLA DE CAMBIOS AL INICIO (CONTROL DE AJUSTES)

| Sección | Qué decía en la entrega anterior | Qué dice ahora (Ajustado) | Por qué cambió (Justificación) |
|---|---|---|---|
| **Portada y Autores** | Presentado individualmente por Jesús Andrés Pérez Fuentes (T00078491). | Presentado en pareja por Jesús Andrés Pérez Fuentes (T00078491) y Cristian David Cardeño Gulloso (T00078446). | Conformación del equipo de trabajo y unificación de los proyectos asignados según directriz del docente. |
| **Sección Previa (Nueva)** | No existía sección de selección previa ni matriz comparativa entre agentes. | **Sección 0: Selección Justificada del Agente (Fase 1)**. Matriz de decisión ponderada (6 criterios, suma 100%), análisis de sensibilidad ($\pm 10$ pts) y justificación de elección frente a *SegmentIQ-Agent*. | Requisito obligatorio de la Fase 1: evaluar formalmente los dos agentes asignados a la pareja y justificar metodológicamente la elección. |
| **Sección 2: Tipo de Agente** | Agente basado en modelo con capacidad de aprendizaje descrito de forma teórica general. | Se formaliza el ciclo explícito en 4 métodos: `perceive()` $\to$ `decide()` $\to$ `act()` $\to$ `learn()`, integrando el elemento de crítica (`ForecastTracker`) y el modelo espacial-temporal Graph WaveNet. | Retroalimentación de la Fase 2: el código debe identificar inequívocamente el ciclo agéntico y la coherencia del estado interno y aprendizaje. |
| **Sección 3: Tipos de Ambiente** | Descripción cualitativa de las 6 dimensiones del ambiente de Russell & Norvig. | Se detalla cómo cada propiedad se emula en el simulador `TrafficEnvironment` (ruido estocástico $\sigma$, fallas de sensor $p$, propagación espacial e impacto en lazo cerrado). | Justificar la validez experimental de las pruebas de Fase 2 y Fase 3 bajo condiciones controladas de incertidumbre. |
| **Sección 5: Entradas y Sensores** | Lista genérica de sensores sin especificación de tratamiento ante datos corruptos o ausentes. | Se especifica el enmascaramiento booleano de valores $0.0$ (faltantes en METR-LA) y el tratamiento de fallas de telemetría y ruidos atípicos. | Garantizar que el agente opere de forma robusta ante fallas masivas de sensores (comprobado en la Fase 3 con 30% de fallas). |
| **Sección 6: Salidas y Actuadores** | Ajustes semafóricos inmediatos sin contemplar la estabilidad del controlador de tránsito. | Se introduce la política de histéresis y enfriamiento (*cooldown* de 30 min): órdenes de plan `PICO_RECURRENTE` o `INCIDENTE` solo ante cambios de estado confirmados. | Evitar oscilaciones continuas (*chattering*) que dañen la coordinación de la red de semáforos real. |
| **Sección 7: Medidas de Desempeño** | *"alcanzando un error absoluto medio (MAE) inferior al 10% respecto de la velocidad real..."* | Se reformula y formaliza matemáticamente como **MAPE** (Mean Absolute Percentage Error) $< 10\%$, complementada con 5 medidas cuantitativas (Recall $\ge 85\%$, FAR $\le 5\%$, Latencia $< 500$ ms, Chattering $\le 10\%$, Resiliencia $\le 5\%$). Se introduce WAPE para horas pico. | Corrección conceptual rigurosa: el error relativo porcentual es MAPE, no MAE. Además, se requiere un conjunto exhaustivo de medidas para la Fase 3. |
| **Sección 8: Componentes IA (IBM)** | Descripción conceptual de las siete áreas de la IA agéntica sin mapeo a módulos. | Mapeo directo y comprobable de cada una de las 7 áreas a los módulos implementados en Python (`agent.py`, `environment.py`, `actuator.py`, `forecasters.py`). | Documentar y atribuir claramente qué parte del razonamiento hace el código propio y qué parte delega en bibliotecas externas. |

---

## SECCIÓN 0: SELECCIÓN JUSTIFICADA DEL AGENTE (FASE 1)

Antes de iniciar el desarrollo técnico, la pareja de trabajo conformada por Jesús Andrés Pérez Fuentes y Cristian David Cardeño Gulloso evaluó los dos agentes inteligentes previamente asignados:
1. **Agente A: TrafficFlow-Agent** (Agente de Predicción y Control de Congestión Vehicular sobre Redes Viales).
2. **Agente B: SegmentIQ-Agent** (Agente de Segmentación Automática de Clientes para E-commerce mediante RFM y Clustering).

### 0.1 Criterios de Evaluación y Ponderación
Se establecieron seis criterios ponderados que suman exactamente el 100 %:
1. **Viabilidad técnica ($w_1 = 25\%$):** Factibilidad de implementación con los conocimientos y tiempo disponibles.
2. **Disponibilidad de datos o entorno ($w_2 = 20\%$):** Acceso a datasets reales y posibilidad de construir un simulador del entorno.
3. **Medibilidad del desempeño ($w_3 = 20\%$):** Capacidad de probar las metas SMART de forma cuantitativa e inobjetable.
4. **Complejidad de IA ($w_4 = 15\%$):** Nivel de razonamiento agéntico genuino (modelo interno, aprendizaje continuo, políticas de acción).
5. **Costo y recursos ($w_5 = 10\%$):** Presupuesto financiero, cómputo y ausencia de consumo de APIs comerciales pagas.
6. **Valor del problema ($w_6 = 10\%$):** Impacto y beneficio social/técnico de la solución planteada.

### 0.2 Matriz de Decisión Ponderada

$$S(\text{agente}) = \sum_{i=1}^6 (w_i \times p_i)$$

| Criterio | Peso ($w$) | Agente A (TrafficFlow-Agent): puntaje (1–5) | Agente A: evidencia | Agente B (SegmentIQ-Agent): puntaje (1–5) | Agente B: evidencia |
|---|:---:|:---:|---|:---:|---|
| **Viabilidad técnica** | 25 % | **5** | El equipo domina PyTorch y series temporales viales; la arquitectura Graph WaveNet y su entrenamiento ya cuentan con pipeline modular funcional. | **4** | K-Means y RFM son directos de implementar con Scikit-Learn, pero simular la retroalimentación de compra del cliente requiere modelar comportamiento humano estocástico complejo. |
| **Disponibilidad de datos o entorno** | 20 % | **5** | Dataset benchmark abierto `METR-LA` disponible localmente (207 sensores, 34.272 registros de 5 min) junto con simulador sintético estocástico funcional (`TrafficEnvironment`). | **4** | Dataset público transaccional disponible (`Online Retail` de UCI), pero no cuenta con un simulador dinámico de interacción ni respuesta a campañas. |
| **Medibilidad del desempeño** | 20 % | **5** | Medidas con fórmula matemática rigurosa (MAPE y MAE a 30 min) verificables objetivamente contra datos reales cada 5 min y probadas en 10 escenarios. | **3** | La meta SMART exige medir "+15% en tasa de respuesta a campañas comerciales en 3 meses", lo cual es inviable de probar experimentalmente sin un e-commerce real activo. |
| **Complejidad de IA** | 15 % | **5** | Agente basado en modelo con aprendizaje: combina Graph WaveNet (convoluciones causales dilatadas + difusión adaptativa sobre grafos), $z$-scores históricos y recalibración continua por deriva. | **3** | Segmentación no supervisada tradicional (K-Means y RFM); opera más como un pipeline analítico mensual por lotes (*batch ETL*) que como un agente reactivo continuo percepción-acción. |
| **Costo y recursos** | 10 % | **4** | 0 COP en APIs de pago; requiere cómputo local con PyTorch (ejecutable en CPU/GPU doméstica en pocos minutos gracias a su diseño compacto). | **5** | 0 COP en APIs; clustering K-Means corre en segundos en cualquier computador con requerimientos mínimos de memoria. |
| **Valor del problema** | 10 % | **5** | Aborda una problemática crítica de interés público en ciudades colombianas (congestión vial urbana, tiempos de viaje, emisiones de $\text{CO}_2$ y control semafórico). | **4** | Solución comercialmente atractiva para focalizar presupuestos de mercadeo en retail, pero de beneficio circunscrito al ámbito privado corporativo. |
| **Puntaje ponderado $S$** | **100 %** | **4.90** | $0.25(5) + 0.20(5) + 0.20(5) + 0.15(5) + 0.10(4) + 0.10(5) = \mathbf{4.90}$ | **3.75** | $0.25(4) + 0.20(4) + 0.20(3) + 0.15(3) + 0.10(5) + 0.10(4) = \mathbf{3.75}$ |

### 0.3 Análisis de Sensibilidad
El criterio más influyente es **Viabilidad técnica ($w_1 = 25\%$)**. Se evaluaron dos variaciones extremas ($\pm 10$ puntos porcentuales) redistribuyendo proporcionalmente la diferencia en los restantes 5 criterios:
- **Variación $+10$ puntos ($w_{\text{Viabilidad}} = 35\%$):** $S(A) = \mathbf{4.91}$ vs. $S(B) = \mathbf{3.78}$. Ventaja para TrafficFlow-Agent: $+1.13$ puntos.
- **Variación $-10$ puntos ($w_{\text{Viabilidad}} = 15\%$):** $S(A) = \mathbf{4.89}$ vs. $S(B) = \mathbf{3.72}$. Ventaja para TrafficFlow-Agent: $+1.17$ puntos.

**Conclusión:** La decisión es completamente robusta e insensible a sesgos de ponderación; TrafficFlow-Agent se mantiene como la mejor alternativa en cualquier escenario.

### 0.4 Justificación de la Elección
Se seleccionó de forma unánime a **TrafficFlow-Agent** (4.90/5.00) sobre **SegmentIQ-Agent** (3.75/5.00). La razón de peso fundamental fue la **medibilidad experimental**: la meta SMART de SegmentIQ-Agent requería medir la tasa de respuesta de clientes reales a campañas de marketing por tres meses, lo cual es inalcanzable en el aula. TrafficFlow-Agent, en contraste, se valida contra observaciones continuas cada 5 minutos en el dataset benchmark `METR-LA` de 207 sensores viales, garantizando una experimentación reproducible, científica y de alto impacto público. Al descartar SegmentIQ-Agent se pierde el abordaje del marketing digital y la relativa simplicidad de algoritmos clásicos de clustering, pero se gana una arquitectura agéntica completa basada en modelo con aprendizaje continuo.

---

## 1. INTRODUCCIÓN Y PLANTEAMIENTO DEL PROBLEMA

La congestión vehicular constituye uno de los principales problemas de movilidad urbana en las ciudades intermedias y grandes de Colombia y el mundo, generando incrementos críticos en los tiempos de desplazamiento, mayor consumo de combustible, emisiones de gases de efecto invernadero y pérdidas económicas multimillonarias. Los sistemas de gestión de tránsito actuales operan casi exclusivamente de manera **reactiva**, ajustando los ciclos semafóricos o emitiendo alertas cuando el embotellamiento ya se ha formado, lo que anula la capacidad preventiva de las autoridades viales.

Para transformar este paradigma hacia una gestión **predictiva y proactiva**, se diseñó **TrafficFlow-Agent**, un agente inteligente concebido para pronosticar la velocidad y el nivel de congestión de un corredor vial con un horizonte de **30 minutos**, utilizando datos históricos y telemetría en tiempo real capturada cada cinco minutos por sensores viales. Un agente inteligente resulta idóneo para este reto porque el entorno vial es continuo, dinámico y parcialmente observable, exigiendo deliberación en tiempo real basada en modelos internos espacio-temporales y la capacidad de accionar de forma autónoma sobre planes semafóricos y paneles de información vial.

---

## 2. TIPO DE AGENTE INTELIGENTE (TAXONOMÍA AIMA)

Según la clasificación formal de Russell y Norvig (*Artificial Intelligence: A Modern Approach*), **TrafficFlow-Agent** se clasifica como un **Agente Basado en Modelo con Capacidad de Aprendizaje** (*Model-based Learning Agent*), estructurado en cuatro componentes formales:

1. **Estado Interno / Modelo del Mundo:** El agente no se limita a reaccionar a la lectura instantánea. Mantiene una memoria de corto plazo de la última hora (12 pasos de 5 min), la matriz de adyacencia del grafo vial ($\mathbf{A} \in \mathbb{R}^{N \times N}$) que modela cómo se propaga la congestión espacialmente, y el `HistoricalProfile` que almacena la velocidad media ($\mu$) y desviación típica ($\sigma$) para cada sensor según el día de la semana y la franja horaria.
2. **Elemento de Rendimiento (*Performance Element*):** La red neuronal espacial-temporal **Graph WaveNet**, que combina convoluciones causales dilatadas (TCN) para capturar dependencias temporales y capas de difusión adaptativa sobre grafos para modelar interacciones entre sensores vecinos.
3. **Elemento de Crítica (*Critic*):** Encapsulado en la clase `ForecastTracker`, el cual compara cada pronóstico emitido hace 30 minutos con la observación real que acaba de percibir el agente, calculando continuamente el MAPE móvil de la última ventana de 24 horas.
4. **Elemento de Aprendizaje (*Learning Element*):** Si la crítica indica que el MAPE a 30 minutos supera la meta del $10\%$, el agente dispara automáticamente un proceso de recalibración (*fine-tuning*) utilizando los datos de los últimos 7 días retenidos en su búfer histórico.

### Ciclo Operativo Explícito (Función del Agente):
El ciclo agéntico se descompone en código en cuatro fases metodológicas:
$$\text{Percepción } [v_t, \text{ts}] \longrightarrow \text{Aprendizaje/Crítica } [e_{t-6}] \longrightarrow \text{Decisión } [\hat{v}_{t+6}, \text{Plan}] \longrightarrow \text{Acción } [\text{TMS}]$$

---

## 3. TIPOS DE AMBIENTE (PROPIEDADES DE RUSSELL & NORVIG)

El entorno del corredor vial exhibe las siguientes seis dimensiones canónicas, las cuales fueron formalmente modeladas en el simulador `TrafficEnvironment`:

| Propiedad | Clasificación | Justificación y Modelado en el Agente |
|---|:---:|---|
| **Observabilidad** | **Parcialmente observable** | El agente capta lecturas únicamente en puntos discretos donde existen lazos inductivos o radares. Existen tramos ciegos e incidentes no registrados de forma directa. El simulador introduce probabilidades estocásticas de fallo de sensor ($p_{\text{fail}} = 2\% \text{ a } 30\%$). |
| **Determinismo** | **Estocástico** | El comportamiento de los conductores, las condiciones climáticas y los incidentes imprevistos introducen incertidumbre intrínseca; estados idénticos no producen resultados idénticos. El simulador inyecta perturbaciones gaussianas $\mathcal{N}(0, \sigma^2)$. |
| **Episodicidad** | **Secuencial** | Cada decisión de ajuste semafórico afecta la fluidez de la vía en intervalos posteriores, y la congestión exhibe inercia temporal y propagación física espacial aguas arriba. |
| **Dinamismo** | **Dinámico** | El tráfico vehicular evoluciona en tiempo real independientemente del tiempo de cómputo del agente. Los pasos temporales avanzan rígidamente cada 5 minutos. |
| **Discretización** | **Continuo** | Las magnitudes de velocidad (mph) y densidad vehicular son continuas, si bien se muestrean a intervalos discretos de 5 minutos. |
| **Número de agentes** | **Agente único** | Desde la perspectiva del control semafórico, opera un agente centralizado, inmerso en un entorno multiagente no coordinado formado por miles de conductores autónomos. |

---

## 4. DESCRIPCIÓN DEL ENTORNO OPERATIVO

El corredor piloto corresponde a una autopista metropolitana instrumentada con detectores fijos distribuidos homogéneamente. Los actores del sistema son:
1. **Conductores:** Usuarios de la vía que reaccionan a las condiciones del tráfico y a los mensajes viales.
2. **Centro de Control de Tránsito:** Operadores humanos que supervisan los planes semafóricos automatizados.
3. **Sistemas Físicos:** Red de semáforos en accesos e intersecciones y Paneles de Mensajería Variable (PMV).

**Restricciones operativas:** Baja latencia en inferencia ($< 500$ ms) para asegurar tiempo de cómputo marginal frente a los 5 minutos del ciclo; robustez ante sensores averiados; y límites físicos de seguridad vial en la extensión de verdes semafóricos (máximo $+30\%$).

---

## 5. ENTRADAS Y SENSORES (PERCEPCIÓN)

El agente percibe su entorno cada cinco minutos a través de los siguientes canales:
* **Velocidad promedio puntual por sensor:** Dato continuo en mph o km/h capturado por 207 lazos inductivos. Los valores cero se tratan estrictamente como datos ausentes (*missing*).
* **Calidad y disponibilidad de la señal:** Máscara booleana que discrimina sensores activos de sensores caídos o desconectados.
* **Marca temporal del sistema:** Fecha y hora exacta (estampilla de tiempo), de la cual se derivan variables cíclicas de hora del día y día de la semana.
* **Historial de tráfico del corredor:** Series temporales de velocidad retenidas en memoria para contrastar contra el perfil típico de largo plazo.
* **Filtro de preprocesamiento:** Detección de valores espurios y normalización Z-score estándar basada en los parámetros $\mu_{\text{train}}$ y $\sigma_{\text{train}}$.

---

## 6. SALIDAS Y ACTUADORES (ACCIÓN)

El agente ejecuta decisiones sobre el entorno a través de dos mecanismos de actuación:
1. **Controlador Semafórico (TMS):** Ajuste de planes semafóricos mediante órdenes estructuradas a la interfaz `TrafficManagementClient`:
   * `PICO_RECURRENTE`: Extensión de verde del $+15\%$ en intersecciones críticas cuando el tráfico es denso pero esperado por la hora.
   * `INCIDENTE`: Extensión de verde del $+30\%$ en accesos alternos y reducción en tramos bloqueados ante incidentes atípicos confirmados.
   * `RESTABLECER_PLAN_BASE`: Retorno a la temporización nominal cuando las velocidades se normalizan ($> 50$ mph).
2. **Paneles de Mensajería Variable (PMV) y Navegación:** Publicación de mensajes de advertencia para inducir desvíos voluntarios en los conductores antes de ingresar a zonas de congestión severa.
3. **Mecanismo de Estabilidad e Histéresis:** Implementación de un período de enfriamiento (*cooldown* de 30 minutos / 6 pasos) para suprimir conmutaciones erráticas entre planes semafóricos contiguos (*chattering*).

---

## 7. MEDIDAS DE DESEMPEÑO SMART FORMALIZADAS

Para superar las ambigüedades de la entrega previa, la medida de desempeño se formalizó matemáticamente y se complementó con indicadores de detección, estabilidad y robustez:

$$\begin{aligned}
\text{M1 — Precisión del Pronóstico a 30 min (SMART):} \quad & \text{MAPE}_{30} = \frac{1}{N \cdot S} \sum_{t=1}^N \sum_{s=1}^S \frac{|\hat{v}_{s, t+6} - v_{s, t+6}|}{v_{s, t+6}} \times 100\% < 10.0\% \\
\text{M2 — Tasa de Detección de Incidentes Atípicos:} \quad & \text{Recall} = \frac{\text{Incidentes detectados}}{\text{Incidentes reales inyectados}} \times 100\% \ge 85.0\% \\
\text{M3 — Tasa de Falsas Alarmas en Flujo Normal:} \quad & \text{FAR} = \frac{\text{Pasos normales clasificados como atípicos}}{\text{Total de pasos normales}} \times 100\% \le 5.0\% \\
\text{M4 — Latencia de Decisión en Tiempo Real:} \quad & \bar{T} = \frac{1}{N} \sum t_{\text{step}} < 500.0\text{ ms por ciclo} \\
\text{M5 — Estabilidad del Actuador Semafórico:} \quad & \text{Chattering} = \frac{\text{Conmutaciones de plan en pasos consecutivos}}{\text{Total de pasos de simulación}} \times 100\% \le 10.0\% \\
\text{M6 — Resiliencia ante Falla Masiva (30% caídos):} \quad & \Delta \text{MAPE} = \text{MAPE}_{30\%\text{ caídos}} - \text{MAPE}_{\text{base}} \le 5.0\%
\end{aligned}$$

> **Ajuste Técnico tras la Fase 3:** En congestión severa ($v < 15$ mph), el denominador reducido del MAPE infla asimétricamente el error porcentual. Se recomienda complementar operativamente con **WAPE** ($\frac{\sum |\hat{v} - v|}{\sum v} \times 100\%$), bajo el cual el error en hora pico desciende por debajo del $8.5\%$.

---

## 8. COMPONENTES DE LA ARQUITECTURA DE IA AGÉNTICA (MARCO IBM)

| Componente IBM | Mapeo en el Software de TrafficFlow-Agent | Descripción Técnica de la Implementación |
|---|---|---|
| **Percepción y manejo de entradas** | `TrafficFlowAgent.perceive()` en `agent.py` | Ingesta de 207 sensores cada 5 min, enmascaramiento booleano de faltantes y normalización Z-score. |
| **Planificación y descomposición** | Orquestador `step()` en `agent.py` | Descomposición secuencial modular: Percepción $\to$ Crítica $\to$ Pronóstico $\to$ Evaluación Z-score $\to$ Acción TMS. |
| **Memoria** | Búfers en `agent.py` y `profile.py` | Memoria de corto plazo (última hora: 12 pasos), memoria histórica semanal (`HistoricalProfile`) y grafo vial `adj_mx`. |
| **Razonamiento y decisión** | `TrafficFlowAgent._assess()` y `model.py` | Inferencia neuronal con Graph WaveNet + clasificación de atípico ($z < -2.0$ en $\ge 15\%$ de sensores) vs. recurrente. |
| **Llamada de herramientas y acción** | `apply_assessment()` en `actuator.py` | Invocación de interfaces `adjust_signal_timings()` y `publish_driver_message()` con filtros de enfriamiento. |
| **Comunicación** | `SimulatedTMS` y logs JSONL | Registro auditable en `outputs/execution_log.jsonl` y reportes JSON para operadores humanos. |
| **Aprendizaje y adaptación** | `ForecastTracker` y `recalibrate()` | Monitoreo del MAPE móvil a 30 min y reentrenamiento adaptativo (*fine-tuning*) ante deriva conceptual ($> 10\%$). |

---

## 9. CONCLUSIONES Y REFERENCIAS

### Conclusiones
El diseño ajustado de **TrafficFlow-Agent** consolida una arquitectura de agente inteligente formal, modular y plenamente reproducible. Al fundamentar la toma de decisiones en un modelo de predicción espacio-temporal sustentado en grafos de conectividad vial y perfiles históricos, el agente supera las limitaciones de los sistemas reactivos tradicionales. Los resultados empíricos derivados de las Fases 2 y 3 corroboran la solidez del diseño: latencia ultra-baja (46.38 ms), detección perfecta de incidentes (Recall 100 %), estabilidad semafórica (Chattering 4.64 %) y resiliencia ante fallas masivas de sensores.

### Referencias Bibliográficas
1. Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Global Edition. Pearson.
2. IBM. (2024). *Components of AI agents*. IBM Think Topics. [https://www.ibm.com/mx-es/think/topics/components-of-ai-agents](https://www.ibm.com/mx-es/think/topics/components-of-ai-agents)
3. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*. International Conference on Learning Representations (ICLR) — METR-LA Benchmark. [https://github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)
4. Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*. Proceedings of IJCAI-19.
5. Jiang, R., et al. (2021). *DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction*. arXiv:2108.09091.
