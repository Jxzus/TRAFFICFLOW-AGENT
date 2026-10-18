# FASE 1 — SELECCIÓN JUSTIFICADA DEL AGENTE
## Matriz de Decisión Ponderada y Análisis Comparativo

- **Estudiantes (Pareja de trabajo):**
  - Jesús Andrés Pérez Fuentes (T00078491) — *TrafficFlow-Agent*
  - Cristian David Cardeño Gulloso (T00078446) — *SegmentIQ-Agent*
- **Curso:** Inteligencia Artificial — Universidad Tecnológica de Bolívar (UTB)
- **Docente:** Edwin Alexander Puertas Del Castillo
- **Fecha:** Octubre de 2026
- **Repositorio:** [https://github.com/Jxzus/TRAFFICFLOW-AGENT](https://github.com/Jxzus/TRAFFICFLOW-AGENT)

---

## 1. Definición de Criterios y Ponderación Inicial (Paso 1)

Para seleccionar de manera objetiva, reproducible y metodológica el agente a desarrollar entre las dos propuestas asignadas, se adoptaron los seis criterios establecidos en la guía académica. Los pesos asignados suman exactamente el **100 % (1.00)**:

1. **Viabilidad técnica ($w_1 = 25\%$):** Evalúa si el agente puede construirse e implementarse con los conocimientos del equipo y el tiempo disponible del semestre.
2. **Disponibilidad de datos o entorno ($w_2 = 20\%$):** Evalúa si existen datasets abiertos reales de alta calidad y si es factible construir un entorno de simulación dinámico.
3. **Medibilidad del desempeño ($w_3 = 20\%$):** Evalúa si las medidas de desempeño SMART pueden probarse de forma matemática, cuantitativa y objetiva en el laboratorio.
4. **Complejidad de IA ($w_4 = 15\%$):** Evalúa si el problema exige razonamiento agéntico genuino (modelo interno, aprendizaje continuo, planificación o utilidad) y no simples agrupamientos estáticos o reglas condicionales.
5. **Costo y recursos ($w_5 = 10\%$):** Evalúa el gasto financiero en hardware, cuotas o consumo de APIs comerciales de pago.
6. **Valor del problema ($w_6 = 10\%$):** Evalúa la utilidad práctica y el impacto real de la solución para un usuario o comunidad identificable.

$$\sum_{i=1}^6 w_i = 0.25 + 0.20 + 0.20 + 0.15 + 0.10 + 0.10 = 1.00 \quad (100\%)$$

---

## 2. Matriz de Decisión Ponderada (Paso 2 y Paso 3)

La evaluación asigna una escala entera de 1 a 5 para cada criterio, acompañada obligatoriamente de su respectiva frase de evidencia técnica:

| Criterio | Peso ($w$) | Agente A (TrafficFlow-Agent): puntaje (1–5) | Agente A: evidencia | Agente B (SegmentIQ-Agent): puntaje (1–5) | Agente B: evidencia |
|---|:---:|:---:|---|:---:|---|
| **Viabilidad técnica** | 25 % | **5** | El equipo domina PyTorch y series temporales viales; la arquitectura Graph WaveNet y su entrenamiento ya cuentan con pipeline modular funcional. | **4** | K-Means y RFM son directos de implementar con Scikit-Learn, pero simular la retroalimentación de compra del cliente requiere modelar comportamiento humano estocástico complejo. |
| **Disponibilidad de datos o entorno** | 20 % | **5** | Dataset benchmark abierto `METR-LA` disponible localmente (207 sensores, 34.272 registros de 5 min) junto con simulador sintético estocástico funcional (`TrafficEnvironment`). | **4** | Dataset público transaccional disponible (`Online Retail` de UCI), pero no cuenta con un simulador dinámico de interacción ni respuesta a campañas. |
| **Medibilidad del desempeño** | 20 % | **5** | Medidas con fórmula matemática rigurosa (MAPE y MAE a 30 min) verificables objetivamente contra datos reales cada 5 min y probadas en 10 escenarios. | **3** | La meta SMART exige medir "+15% en tasa de respuesta a campañas comerciales en 3 meses", lo cual es inviable de probar experimentalmente sin un e-commerce real activo. |
| **Complejidad de IA** | 15 % | **5** | Agente basado en modelo con aprendizaje: combina Graph WaveNet (convoluciones causales dilatadas + difusión adaptativa sobre grafos), $z$-scores históricos y recalibración continua por deriva. | **3** | Segmentación no supervisada tradicional (K-Means y RFM); opera más como un pipeline analítico mensual por lotes (*batch ETL*) que como un agente reactivo continuo percepción-acción. |
| **Costo y recursos** | 10 % | **4** | 0 COP en APIs de pago; requiere cómputo local con PyTorch (ejecutable en CPU/GPU doméstica en pocos minutos gracias a su diseño compacto). | **5** | 0 COP en APIs; clustering K-Means corre en segundos en cualquier computador con requerimientos mínimos de memoria. |
| **Valor del problema** | 10 % | **5** | Aborda una problemática crítica de interés público en ciudades colombianas (congestión vial urbana, tiempos de viaje, emisiones de $\text{CO}_2$ y control semafórico). | **4** | Solución comercialmente atractiva para focalizar presupuestos de mercadeo en retail, pero de beneficio circunscrito al ámbito privado corporativo. |
| **Puntaje ponderado $S$** | **100 %** | **4.90** | $S(A) = 0.25(5) + 0.20(5) + 0.20(5) + 0.15(5) + 0.10(4) + 0.10(5) = \mathbf{4.90}$ | **3.75** | $S(B) = 0.25(4) + 0.20(4) + 0.20(3) + 0.15(3) + 0.10(5) + 0.10(4) = \mathbf{3.75}$ |

---

## 3. Análisis de Sensibilidad (Paso 4)

El criterio con mayor ponderación es **Viabilidad técnica** ($w_1 = 25\%$). Para verificar la robustez de la decisión ante eventuales discrepancias o sesgos en la asignación de pesos, se varió dicho peso en **$\pm 10$ puntos porcentuales**, redistribuyendo proporcionalmente la diferencia en los restantes 5 criterios:

### Caso A: Incremento de peso (+10 puntos $\to w_1 = 35\%$)
Al dar máxima prioridad a la facilidad y conocimiento técnico previo:
- $w_{\text{Viabilidad}} = 0.35$
- Los demás criterios absorben el ajuste proporcional: $w_{\text{Datos}}=0.1733$, $w_{\text{Medibilidad}}=0.1733$, $w_{\text{Complejidad}}=0.1300$, $w_{\text{Costo}}=0.0867$, $w_{\text{Valor}}=0.0867$.
- **Cálculo:**
  $$S(A) = 0.35(5) + 0.1733(5) + 0.1733(5) + 0.1300(5) + 0.0867(4) + 0.0867(5) = \mathbf{4.91}$$
  $$S(B) = 0.35(4) + 0.1733(4) + 0.1733(3) + 0.1300(3) + 0.0867(5) + 0.0867(4) = \mathbf{3.78}$$
- **Resultado:** TrafficFlow-Agent mantiene una ventaja contundente de $+1.13$ puntos.

### Caso B: Reducción de peso (-10 puntos $\to w_1 = 15\%$)
Al reducir el peso de la viabilidad técnica y priorizar la medibilidad, complejidad y costo:
- $w_{\text{Viabilidad}} = 0.15$
- Criterios restantes ajustados proporcionalmente: $w_{\text{Datos}}=0.2267$, $w_{\text{Medibilidad}}=0.2267$, $w_{\text{Complejidad}}=0.1700$, $w_{\text{Costo}}=0.1133$, $w_{\text{Valor}}=0.1133$.
- **Cálculo:**
  $$S(A) = 0.15(5) + 0.2267(5) + 0.2267(5) + 0.1700(5) + 0.1133(4) + 0.1133(5) = \mathbf{4.89}$$
  $$S(B) = 0.15(4) + 0.2267(4) + 0.2267(3) + 0.1700(3) + 0.1133(5) + 0.1133(4) = \mathbf{3.72}$$
- **Resultado:** TrafficFlow-Agent amplía su ventaja a $+1.17$ puntos.

**Conclusión del análisis de sensibilidad:** La decisión es perfectamente estable e invariable. El orden de preferencia no se invierte bajo ningún escenario, demostrando que la superioridad metodológica de TrafficFlow-Agent no depende de una calibración arbitraria de pesos.

---

## 4. Redacción de la Justificación de Selección (Paso 5 — Máximo una página)

La pareja de trabajo seleccionó de manera unánime y fundamentada a **TrafficFlow-Agent (Agente de Predicción y Control de Congestión Vehicular)** frente a **SegmentIQ-Agent (Agente de Segmentación de Clientes)**, tras obtener un puntaje ponderado de **4.90 sobre 5.00** frente a **3.75 sobre 5.00** en la matriz de decisión.

Las razones técnicas determinantes de la elección son:

1. **Medibilidad Objetiva y Validación Experimental:**
   La principal debilidad identificada en SegmentIQ-Agent radica en su medida de desempeño SMART: medir un *"incremento del 15% en la tasa de respuesta a campañas de marketing dirigidas durante tres meses"* es impracticable en el laboratorio sin una empresa real activa ejecutando campañas masivas de pauta o correo. En contraste, TrafficFlow-Agent opera sobre una métrica de ingeniería de tráfico universalmente contrastable ($\text{MAPE} < 10\%$), cuya verdad de terreno (*ground truth*) se actualiza cada 5 minutos mediante sensores físicos reales de velocidad en autopista (`METR-LA`), permitiendo una validación científica inmediata, repetible y automatizada.

2. **Genuina Arquitectura de Inteligencia Artificial Agéntica:**
   SegmentIQ-Agent, pese a utilizar K-Means y variables RFM, encaja conceptualmente más como un pipeline tradicional de analítica de datos por lotes (*batch analytics*) que como un agente autónomo interactivo. Por el contrario, TrafficFlow-Agent encarna en su totalidad la arquitectura del **Agente Basado en Modelo con Aprendizaje** descrita por Russell y Norvig:
   - Percibe un entorno dinámico continuo cada 5 minutos.
   - Mantiene un estado interno enriquecido (grafo de conectividad vial y memoria de corto/largo plazo).
   - Pronostica a 30 minutos combinando redes neuronales espaciotemporales (Graph WaveNet) con perfiles históricos de $z$-score.
   - Actúa de forma cerrada sobre controladores semafóricos y paneles de información vial aplicando políticas de estabilización.
   - Aprende y se recalibra ante derivas conceptuales del tráfico cuando el componente de crítica detecta degradación del error.

3. **Disponibilidad de Datos y Entorno Operativo:**
   TrafficFlow-Agent cuenta con el dataset canónico `METR-LA` (207 detectores viales, 34.272 pasos temporales en Los Ángeles) y un simulador estocástico validado (`TrafficEnvironment`), lo que permite someter al agente a escenarios extremos de fallas de sensores, ruidos meteorológicos e incidentes imprevistos.

### ¿Qué se pierde al descartar SegmentIQ-Agent?
Al descartar SegmentIQ-Agent se prescinde de abordar el dominio de la analítica de clientes y marketing digital, así como de la relativa simplicidad matemática e implementación ligera del agrupamiento no supervisado, el cual requiere menor potencia computacional que el entrenamiento tensorial con PyTorch. No obstante, el equipo concluyó que dicha facilidad técnica inicial no compensaba la falta de un entorno interactivo cerrado ni el riesgo metodológico de no poder validar empíricamente la efectividad de las decisiones comerciales del agente en las Fases 2 y 3.

Por consiguiente, **TrafficFlow-Agent** representa la propuesta con mayor rigor de ingeniería, mejor alineación con los objetivos de la materia y viabilidad experimental demostrada.
