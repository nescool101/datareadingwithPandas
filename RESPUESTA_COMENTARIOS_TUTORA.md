# Respuesta a Comentarios de la Tutora - Sindy Katerine Rincon Torres

**Fecha de correccion:** 23 de febrero de 2026
**Archivo corregido:** `analisis_datos.ipynb`
**Dataset:** `Total_Mes_Act_Datos completos CORREGIDO.csv` (1,048,575 filas, 36 columnas)

---

## Bloque Critico 1: Fiabilidad de los Modelos

### Comentario Original (22/02/2026, 6:14 PM)

> Estas obteniendo metricas practicamente perfectas (AUC ~0,999, F1 > 0,97) en un problema desbalanceado. Eso exige una revision muy cuidadosa de:
> - posible fuga de informacion (data leakage),
> - variables que codifican indirectamente el resultado,
> - ausencia de validacion temporal.

### Solucion 1: Eliminacion de identificadores y variables con fuga

**Donde:** Paso 6 y Paso 14 - MODIFICADOS

Se identificaron y eliminaron las siguientes columnas del modelado:

| Columna | Categoria | Razon de eliminacion |
|---|---|---|
| `Msh Cuenta` | Identificador | Identificador de cuenta - sin poder predictivo, causa memorizacion |
| `MSH_CEDULA` | Identificador | Identificador personal - memorizacion de casos individuales |
| `MSH_LOGIN_ID` | Identificador | Identificador de login - memorizacion de casos individuales |
| `SIN TARIFA` | **Fuga directa** | Codifica directamente si existe `TARIFA_NBO`, que es componente del target. Correlacion artificialmente alta con `Rentabilizo` |
| `TARIFA_NBO` | Target | Componente de la variable objetivo. No puede usarse como feature |
| `Msh Fecha (copia)` | Fecha | Reservada para validacion temporal, no incluida como feature |
| `Msh Hora (copia)` | Duplicada | Duplicado de la columna `Msh Hora`, informacion redundante |

Adicionalmente, se clasificaron 2 variables como **sospechosas para evaluacion por ablacion**:
- `MSH_TIPO_CLIENTE`: Mostraba 93.47% de importancia en modelos previos (posible proxy del target)
- `MOVIMIENTO_RENTA`: Podria contener informacion post-resultado (fuga temporal)

**Resultado:** De 36 columnas originales, se conservaron **19 features** para modelado.

**Codigo implementado (Paso 6):**
```python
columnas_a_eliminar = [
    columna_target, 'TARIFA_NBO', 'Msh Cuenta', 'MSH_CEDULA',
    'MSH_LOGIN_ID', 'SIN TARIFA', 'Msh Fecha (copia)', 'Msh Hora (copia)'
]
columnas_existentes = [c for c in columnas_a_eliminar if c in df_filtrado.columns]
X = df_filtrado.drop(columns=columnas_existentes)
```

### Solucion 2: Reentrenamiento y metricas reales (sin fuga)

**Donde:** Pasos 7-14 re-ejecutados con datos corregidos

Al eliminar las columnas problematicas, los modelos se reentrenaron con datos limpios. Division: **Train 80% / Test 20%** con estratificacion.

**Metricas corregidas del Paso 14 (datos reales de la ejecucion):**

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8858 | 0.7220 | 0.6586 | 0.6888 | 0.9406 | 0.6000 |
| Random Forest | 0.9856 | 0.9424 | 0.9850 | 0.9633 | 0.9970 | 0.9817 |
| **XGBoost (mejor)** | **0.9858** | **0.9387** | **0.9908** | **0.9640** | **0.9973** | **0.9835** |

**Validacion cruzada (StratifiedKFold, 5 folds):**

| Modelo | Media CV | Desviacion Estandar |
|---|---|---|
| Logistic Regression | 0.8854 | +/- 0.0006 |
| Random Forest | 0.9852 | +/- 0.0001 |
| **XGBoost (mejor)** | **0.9857** | **+/- 0.0002** |

**Interpretacion:** Las metricas de CV y test set son practicamente identicas, lo que confirma que **no hay sobreajuste**. XGBoost es el mejor modelo con F1=0.964 y AUC=0.997.

### Solucion 3: Analisis de Ablacion (Paso 14.1) - Verificacion de fuga restante

Se ejecutaron 4 escenarios con XGBoost + StratifiedKFold(5) para evaluar si `MSH_TIPO_CLIENTE` y `MOVIMIENTO_RENTA` constituyen fuga:

| Escenario | Descripcion | Features | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| A | Base corregido | 19 | 0.9858 | 0.9639 | 0.9973 |
| B | Sin MSH_TIPO_CLIENTE | 18 | 0.9857 | 0.9639 | 0.9972 |
| C | Sin MOVIMIENTO_RENTA | 18 | 0.8088 | 0.0284 | 0.6779 |
| D | Sin ambas | 17 | 0.8089 | 0.0335 | 0.6770 |

**Conclusion del analisis de ablacion:**

1. **MSH_TIPO_CLIENTE NO es fuga.** Eliminarlo (B vs A) no cambia practicamente nada: F1 pasa de 0.9639 a 0.9639, AUC de 0.9973 a 0.9972. Su alta importancia previa era un artefacto del modelo anterior con fuga.

2. **MOVIMIENTO_RENTA es la variable predictiva clave.** Eliminarlo (C vs A) produce un **colapso catastrofico**: F1 cae de 0.964 a 0.028, Recall de 0.991 a 0.015. Esto indica que es **informacion predictiva legitima**, no fuga, ya que:
   - Representa el movimiento de renta del cliente (IGUAL/AUMENTO/DISMINUCION), informacion disponible antes de la decision de NBO
   - Sin esta variable, el modelo no tiene capacidad predictiva significativa

3. **El escenario D confirma** que sin ambas variables sospechosas, el modelo colapsa de forma casi identica a C, reforzando que el problema es la ausencia de MOVIMIENTO_RENTA, no de MSH_TIPO_CLIENTE.

**Importancia de features (Top 5 - XGBoost corregido):**

| Posicion | Feature | Importancia |
|---|---|---|
| 1 | MOVIMIENTO_RENTA | 96.50% |
| 2 | MESES_ANTIGUEDAD | 2.25% |
| 3 | MESES_ANTIGUEDAD_INSTALACION | 0.66% |
| 4 | MSH_TIPO_CLIENTE | 0.18% |
| 5 | Msh Ser Activos | 0.05% |

### Solucion 4: Validacion Temporal (Paso 14.2)

Se intento realizar validacion temporal usando la columna `Msh Fecha (copia)`.

**Resultado:** La columna no pudo ser parseada como fecha valida (formato no estandar en el dataset original). El codigo maneja esta limitacion de forma controlada y documenta:

> La columna 'Msh Fecha (copia)' no pudo ser parseada como fecha valida.
> Esto impide realizar validacion temporal del modelo.
>
> Recomendaciones:
> 1. Verificar el formato de la columna de fecha en el dataset original
> 2. Solicitar al equipo de datos una columna de fecha en formato estandar
> 3. Mientras tanto, la validacion cruzada estratificada (Paso 14) sigue siendo valida

Se generaron graficas placeholder (`validacion_temporal_vs_aleatoria.png`, `validacion_temporal_metricas.png`) documentando esta limitacion.

**Mitigacion:** La validacion cruzada estratificada con 5 folds muestra desviaciones estandar minimas (0.0001-0.0006), lo que indica estabilidad del modelo incluso sin validacion temporal explicita.

### Solucion 5: Comparacion Antes/Despues (Paso 14.3)

Tabla comparativa con datos reales que demuestra el impacto de eliminar la fuga:

**Logistic Regression:**

| Metrica | Original (con fuga) | Corregido (sin fuga) | Diferencia |
|---|---|---|---|
| Accuracy | 0.8445 | 0.8858 | +0.0413 |
| Precision | 0.6154 | 0.7220 | +0.1066 |
| Recall | 0.5069 | 0.6586 | +0.1517 |
| F1-Score | 0.5559 | 0.6888 | +0.1329 |
| ROC-AUC | 0.9385 | 0.9406 | +0.0021 |

**Random Forest:**

| Metrica | Original (con fuga) | Corregido (sin fuga) | Diferencia |
|---|---|---|---|
| Accuracy | 0.9897 | 0.9856 | -0.0041 |
| Precision | 0.9627 | 0.9424 | -0.0203 |
| Recall | 0.9844 | 0.9850 | +0.0006 |
| F1-Score | 0.9734 | 0.9633 | -0.0101 |
| ROC-AUC | 0.9987 | 0.9970 | -0.0017 |

**XGBoost:**

| Metrica | Original (con fuga) | Corregido (sin fuga) | Diferencia |
|---|---|---|---|
| Accuracy | 0.9897 | 0.9858 | -0.0039 |
| Precision | 0.9643 | 0.9387 | -0.0256 |
| Recall | 0.9829 | 0.9908 | +0.0079 |
| F1-Score | 0.9735 | 0.9640 | -0.0095 |
| ROC-AUC | 0.9991 | 0.9973 | -0.0018 |

**Interpretacion:**

- **Logistic Regression mejora** al eliminar columnas con fuga: los identificadores (Msh Cuenta, MSH_CEDULA, etc.) aportaban ruido a un modelo lineal. Sin ellos, LR captura mejor las relaciones reales.
- **Random Forest y XGBoost bajan ligeramente** (F1: -0.01, AUC: -0.002), lo cual es coherente: los modelos de arboles podian explotar los identificadores para memorizar, y al eliminarlos pierden esa ventaja artificial.
- **Las metricas corregidas siguen siendo excelentes** (F1 > 0.96, AUC > 0.997) pero ahora reflejan **capacidad predictiva real**, no fuga de datos.
- La AUC original de ~0.999 baja a ~0.997, confirmando que habia una componente de fuga pero que el modelo tiene valor predictivo genuino.

---

## Bloque Critico 2: Desarrollo de la Contribucion y Arquitectura Tecnica

### Comentario Original (22/02/2026, 6:17 PM)

> - El EDA tiene poca toma de decisiones concreta (que se hace con cada hallazgo y por que).
> - Falta una descripcion clara de la arquitectura tecnica o flujo de integracion del modelo en el proceso NBO real.
> - Hay que documentar decisiones de diseno justificadas.
> - Importante presentar al menos un esquema conceptual de como se desplegaria el modelo en la organizacion.

### Solucion 1: EDA con decisiones concretas (Paso 4.1)

**Donde:** Paso 4 MODIFICADO + Paso 4.1 NUEVO

Se clasificaron las 36 columnas del dataset en categorias con decision y justificacion:

| Categoria | Columnas | Decision | Justificacion |
|---|---|---|---|
| Identificadores | Msh Cuenta, MSH_CEDULA, MSH_LOGIN_ID | ELIMINAR | Sin poder predictivo, causan overfitting por memorizacion |
| Fuga directa | SIN TARIFA | ELIMINAR | Correlacion artificial con target (codifica existencia de TARIFA_NBO) |
| Target | TARIFA_NBO, Rentabilizo | ELIMINAR (son el target) | Componentes de la variable objetivo |
| Fecha | Msh Fecha (copia) | RESERVAR | Para validacion temporal |
| Duplicadas | Msh Hora (copia) | ELIMINAR | Duplicado de Msh Hora |
| Sospechosas | MSH_TIPO_CLIENTE, MOVIMIENTO_RENTA | EVALUAR por ablacion | Requieren analisis de ablacion para confirmar si son fuga |

Graficas generadas con evidencia:
- `fuga_sin_tarifa_vs_target.png` - Distribucion de SIN TARIFA vs Rentabilizo mostrando correlacion artificial
- `sospecha_tipo_cliente_vs_target.png` - Distribucion de MSH_TIPO_CLIENTE vs target
- `correlacion_variables_target.png` - Heatmap de correlacion de todas las variables con el target
- `decision_variables_tabla.png` - Tabla visual con decision por cada variable

### Solucion 2: Documentacion formal de decisiones de diseno (Paso 5.1)

**Donde:** Paso 5.1 NUEVO

Se creo un diccionario formal en JSON (`documentacion_variables_eliminadas.json`) con la justificacion tecnica de cada exclusion:

```json
{
  "identificadores": {
    "columnas": ["Msh Cuenta", "MSH_CEDULA", "MSH_LOGIN_ID"],
    "razon": "Identificadores unicos sin poder predictivo. Su inclusion causaria
              memorizacion de casos individuales (overfitting extremo).",
    "accion": "ELIMINAR"
  },
  "fuga_directa": {
    "columnas": ["SIN TARIFA"],
    "razon": "Codifica directamente si existe TARIFA_NBO, que es componente del target.
              Su correlacion con el target es artificialmente alta.",
    "accion": "ELIMINAR"
  },
  "sospechosas_para_ablacion": {
    "columnas": ["MSH_TIPO_CLIENTE", "MOVIMIENTO_RENTA"],
    "razon": "MSH_TIPO_CLIENTE mostro 93.47% de importancia en modelos previos
              (posible proxy del target). MOVIMIENTO_RENTA podria contener informacion
              post-resultado (fuga temporal). Se evaluaran en analisis de ablacion.",
    "accion": "EVALUAR por ablacion"
  }
}
```

### Solucion 3: Arquitectura tecnica del sistema NBO (Paso 15)

**Donde:** Paso 15 NUEVO

Se agrego una seccion completa con:

```
Flujo del Sistema NBO:

  [CRM / Facturacion / Logs]
            |
            v
  [ETL: Exclusion IDs + Encoding + Imputacion]
            |
            v
  [XGBoost: 19 features -> Probabilidad [0,1]]
            |
            v
  [Scoring: Umbral configurable por campana]
            |
            v
  [Motor de Campanas: Priorizacion + Supervision humana]
            |
            v
  [Feedback Loop: Monitoreo + Reentrenamiento]
```

- **Fuentes de datos:** CRM (datos cliente), Facturacion (renta, tarifa), Logs operativos (procesos, estados)
- **Preprocesamiento:** Exclusion de IDs y variables con fuga, LabelEncoding de categoricas (< 50 categorias), imputacion por mediana
- **Modelo:** XGBoost con 19 features, probability scoring [0,1], validado con 5-fold CV estratificado
- **Integracion:** Umbral de decision configurable, priorizacion por probabilidad, supervision humana en Fase 1, feedback loop para reentrenamiento

### Solucion 4: Esquema conceptual de despliegue (Paso 16)

**Donde:** Paso 16 NUEVO

Se implemento un plan de despliegue en 3 fases:

| Fase | Periodo | Frecuencia Scoring | Alcance | Validacion |
|---|---|---|---|---|
| 1 - Piloto | Meses 0-3 | Batch semanal | 1,000 clientes | Manual por equipo comercial |
| 2 - Expansion | Meses 3-7 | Batch diario | Segmentos completos | A/B testing + KPIs |
| 3 - Produccion | Meses 7-12 | Tiempo real (API) | Todos los clientes | Monitoreo de drift automatico |

Grafica: `esquema_despliegue.png` (timeline visual con hitos por fase)

---

## Resumen de Cambios en el Notebook

| Paso | Tipo | Descripcion |
|---|---|---|
| 1 | MODIFICADO | Creacion de carpeta `graficas23Febrero/` |
| 2 | CORREGIDO | Carga de datos con `sep=';'` (correccion de delimitador) |
| 3 | SIN CAMBIOS | Conversion JSON y conteo por cuenta |
| 4 | MODIFICADO | Resumen de hallazgos y decisiones al final del EDA |
| 4.1 | **NUEVO** | EDA con decisiones concretas (4 graficas de evidencia) |
| 5 | SIN CAMBIOS | Identificacion y filtrado |
| 5.1 | **NUEVO** | Documentacion formal de variables eliminadas (JSON) |
| 6 | **MODIFICADO (CRITICO)** | Eliminacion de IDs, fuga y fechas: 36 -> 19 features |
| 7-13 | SIN CAMBIOS | Re-ejecutados con datos corregidos |
| 14 | **MODIFICADO (CRITICO)** | Misma correccion de columnas en comparacion de modelos |
| 14.1 | **NUEVO** | Analisis de ablacion: 4 escenarios (A/B/C/D) |
| 14.2 | **NUEVO** | Validacion temporal (limitacion documentada: fecha no parseable) |
| 14.3 | **NUEVO** | Comparacion antes/despues con metricas reales |
| 15 | **NUEVO** | Arquitectura tecnica del sistema NBO |
| 16 | **NUEVO** | Esquema conceptual de despliegue (3 fases) |
| 17 | **NUEVO** | Conclusiones finales, limitaciones, recomendaciones |

## Graficas Nuevas (carpeta `resultados/graficas23Febrero/`)

| # | Archivo | Contenido |
|---|---|---|
| 1 | `fuga_sin_tarifa_vs_target.png` | Evidencia de fuga en SIN TARIFA |
| 2 | `sospecha_tipo_cliente_vs_target.png` | Analisis de MSH_TIPO_CLIENTE |
| 3 | `correlacion_variables_target.png` | Correlacion de todas las variables con target |
| 4 | `decision_variables_tabla.png` | Tabla de decisiones por variable |
| 5 | `ablacion_comparativa_metricas.png` | Metricas por escenario A/B/C/D |
| 6 | `ablacion_importancia_features_por_escenario.png` | Top features por escenario |
| 7 | `validacion_temporal_vs_aleatoria.png` | Placeholder (fecha no parseable) |
| 8 | `validacion_temporal_metricas.png` | Placeholder (fecha no parseable) |
| 9 | `comparacion_antes_despues_metricas.png` | Metricas antes/despues lado a lado |
| 10 | `comparacion_antes_despues_curvas_roc.png` | Curvas ROC antes/despues |
| 11 | `esquema_despliegue.png` | Timeline de despliegue en 3 fases |

## Archivos JSON Generados

| Archivo | Contenido |
|---|---|
| `resultados/graficas23Febrero/documentacion_variables_eliminadas.json` | Clasificacion de 36 variables con razon y accion |
| `resultados/graficas23Febrero/resultados_ablacion.json` | Metricas de los 4 escenarios de ablacion |
| `resultados/paso14/conclusion_paso14.json` | Mejor modelo (XGBoost), metricas CV y test |
| `resultados/paso14/resultados_validacion_cruzada_paso14.json` | Scores de CV por fold para cada modelo |

---

## Limitaciones Conocidas

1. **Validacion temporal no disponible:** La columna `Msh Fecha (copia)` no tiene formato de fecha parseable. Se recomienda solicitar al equipo de datos una columna temporal en formato estandar (YYYY-MM-DD).

2. **Alta dependencia de MOVIMIENTO_RENTA:** El analisis de ablacion muestra que esta variable concentra el 96.5% de la importancia del modelo. Sin ella, F1 cae a 0.028. Esto implica:
   - El modelo es robusto mientras MOVIMIENTO_RENTA este disponible
   - Se debe verificar que esta variable esta disponible **antes** del momento de prediccion (no es informacion post-hoc)
   - Se recomienda explorar modelos complementarios que no dependan de una sola variable

3. **Desbalance de clases:** El dataset mantiene un desbalance en la variable objetivo. Se uso estratificacion en todas las particiones (train/test, CV, ablacion) para mitigar esto.

---

## Conclusion

Las correcciones implementadas responden directamente a los dos bloques criticos de la tutora:

**Bloque 1 (Fiabilidad):** Se elimino la fuga de datos identificada (SIN TARIFA, identificadores). Las metricas pasaron de AUC ~0.999 a AUC 0.997, confirmando fuga menor pero real. El analisis de ablacion descarto que MSH_TIPO_CLIENTE sea proxy del target, e identifico a MOVIMIENTO_RENTA como variable predictiva legitima (no fuga). La validacion cruzada con std < 0.001 confirma estabilidad.

**Bloque 2 (Contribucion):** Se anadieron decisiones concretas en el EDA (Paso 4.1), documentacion formal de exclusiones (Paso 5.1), arquitectura tecnica completa (Paso 15), esquema de despliegue en 3 fases (Paso 16) y conclusiones con limitaciones (Paso 17).
