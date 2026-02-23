# Respuesta a Comentarios de la Tutora - Sindy Katerine Rincon Torres

**Fecha de correccion:** 23 de febrero de 2026
**Archivo corregido:** `analisis_datos.ipynb`

---

## Bloque Critico 1: Fiabilidad de los Modelos

### Comentario Original (22/02/2026, 6:14 PM)

> Estas obteniendo metricas practicamente perfectas (AUC ~0,999, F1 > 0,97) en un problema desbalanceado. Eso exige una revision muy cuidadosa de:
> - posible fuga de informacion (data leakage),
> - variables que codifican indirectamente el resultado,
> - ausencia de validacion temporal.

### Resoluciones Implementadas

#### 1. Eliminacion de identificadores y variables con fuga

**Donde:** Paso 6 (Cell 17) y Paso 14 (Cell 55) - MODIFICADOS

Se identificaron y eliminaron las siguientes columnas del modelado:

| Columna | Razon de eliminacion |
|---|---|
| `Msh Cuenta` | Identificador de cuenta - sin poder predictivo real |
| `MSH_CEDULA` | Identificador personal - memorizacion de casos |
| `MSH_LOGIN_ID` | Identificador de login - memorizacion de casos |
| `SIN TARIFA` | **Fuga directa**: codifica si existe `TARIFA_NBO` (componente del target) |
| `TARIFA_NBO` | Componente de la variable objetivo |
| `Msh Fecha (copia)` | Reservada para validacion temporal |
| `Msh Hora (copia)` | Duplicado de `Msh Hora` |

**Codigo critico (Paso 6):**
```python
columnas_a_eliminar = [
    columna_target, 'TARIFA_NBO', 'Msh Cuenta', 'MSH_CEDULA',
    'MSH_LOGIN_ID', 'SIN TARIFA', 'Msh Fecha (copia)', 'Msh Hora (copia)'
]
columnas_existentes = [c for c in columnas_a_eliminar if c in df_filtrado.columns]
X = df_filtrado.drop(columns=columnas_existentes)
```

#### 2. Reentrenamiento y reevaluacion bajo restricciones

**Donde:** Pasos 7-14 se re-ejecutan automaticamente con datos corregidos

Al corregir los Pasos 6 y 14, todos los modelos downstream (Logistic Regression, Random Forest, XGBoost) se reentrenan sin las columnas problematicas. Las metricas resultantes seran menores pero **reflejan capacidad predictiva real**.

#### 3. Seccion de validacion critica

**Donde:** Pasos 14.1, 14.2 y 14.3 (Cells 65-70) - NUEVOS

Se anadieron tres secciones de validacion critica:

**Paso 14.1 - Analisis de Ablacion (Cell 66):**
- 4 escenarios con XGBoost + StratifiedKFold(5):
  - A: Modelo corregido base (sin IDs ni SIN TARIFA)
  - B: A + eliminar MSH_TIPO_CLIENTE
  - C: A + eliminar MOVIMIENTO_RENTA
  - D: A + eliminar ambas (solo variables seguras)
- Graficas: `ablacion_comparativa_metricas.png`, `ablacion_importancia_features_por_escenario.png`

**Paso 14.2 - Validacion Temporal (Cell 68):**
- Parseo de `Msh Fecha (copia)` como datetime
- Division temporal: 80% primero (train) / 20% ultimo (test)
- Entrenamiento de XGBoost y Random Forest en train temporal
- Comparacion de metricas temporales vs aleatorias
- Manejo gracioso si la fecha no es parseable (documenta la limitacion)
- Graficas: `validacion_temporal_vs_aleatoria.png`, `validacion_temporal_metricas.png`

**Paso 14.3 - Comparacion Antes/Despues (Cell 70):**
- Tabla lado a lado: metricas originales (con fuga) vs corregidas (sin fuga)
- Curvas ROC comparativas
- Analisis narrativo explicando por que metricas menores son MEJORES
- Graficas: `comparacion_antes_despues_metricas.png`, `comparacion_antes_despues_curvas_roc.png`

---

## Bloque Critico 2: Desarrollo de la Contribucion y Arquitectura Tecnica

### Comentario Original (22/02/2026, 6:17 PM)

> - El EDA tiene poca toma de decisiones concreta (que se hace con cada hallazgo y por que).
> - Falta una descripcion clara de la arquitectura tecnica o flujo de integracion del modelo en el proceso NBO real.
> - Hay que documentar decisiones de diseno justificadas.
> - Importante presentar al menos un esquema conceptual de como se desplegaria el modelo en la organizacion.

### Resoluciones Implementadas

#### 1. EDA con decisiones concretas

**Donde:** Paso 4 (Cell 9) MODIFICADO + Paso 4.1 (Cells 10-11) NUEVO

- Se agrego un resumen de hallazgos clave y decisiones tomadas al final del EDA original
- Se creo el Paso 4.1 que:
  - Clasifica las 36 columnas en: identificadores, fuga, sospechosas, validas
  - Calcula correlacion de `SIN TARIFA` y `MSH_TIPO_CLIENTE` con el target
  - Genera graficas de evidencia de fuga y sospecha
  - Produce una tabla resumen con: nombre, tipo, %nulos, decision, justificacion por cada variable
  - Graficas: `fuga_sin_tarifa_vs_target.png`, `sospecha_tipo_cliente_vs_target.png`, `correlacion_variables_target.png`, `decision_variables_tabla.png`

#### 2. Documentacion de decisiones de diseno

**Donde:** Paso 5.1 (Cells 14-15) NUEVO + Paso 17 (Cell 74) NUEVO

- Paso 5.1 crea un diccionario formal de exclusiones con justificacion tecnica para cada variable eliminada, guardado como JSON
- Paso 17 incluye tabla de justificaciones de diseno y razonamiento para cada decision

#### 3. Arquitectura tecnica del sistema NBO

**Donde:** Paso 15 (Cell 71) NUEVO

Se agrego una seccion completa de arquitectura con:
- Diagrama ASCII del flujo: Fuentes de Datos -> ETL/Preprocesamiento -> Modelo ML -> Scoring -> Motor de Campanas
- Descripcion de fuentes: CRM, Facturacion, Logs operativos
- Pipeline de preprocesamiento: exclusion de IDs, encoding, imputacion
- Modelo de scoring: XGBoost, probabilidad [0,1], validacion cruzada y temporal
- Integracion con campanas: umbral configurable, priorizacion, supervision humana, feedback loop

#### 4. Esquema conceptual de despliegue

**Donde:** Paso 16 (Cells 72-73) NUEVO

Se implemento un esquema visual con 3 fases:
- **Fase 1 - Piloto** (Meses 0-3): Scoring batch semanal, muestra de 1,000 clientes, validacion manual
- **Fase 2 - Expansion** (Meses 3-7): Scoring batch diario, integracion CRM, A/B testing
- **Fase 3 - Produccion** (Meses 7-12): Scoring en tiempo real (API), reentrenamiento automatico, monitoreo de drift
- Grafica: `esquema_despliegue.png`

---

## Resumen de Cambios en el Notebook

| Paso | Tipo | Descripcion |
|---|---|---|
| 1 | MODIFICADO | Creacion de carpeta `graficas23Febrero/` |
| 2-3 | SIN CAMBIOS | Carga de datos y conversion JSON |
| 4 | MODIFICADO | Resumen de hallazgos y decisiones al final del EDA |
| 4.1 | **NUEVO** | EDA con decisiones concretas (4 graficas) |
| 5 | SIN CAMBIOS | Identificacion y filtrado |
| 5.1 | **NUEVO** | Documentacion formal de variables eliminadas |
| 6 | **MODIFICADO (CRITICO)** | Eliminacion de IDs, fuga y fechas del modelado |
| 7-13 | SIN CAMBIOS | Re-ejecutados con datos corregidos |
| 14 | **MODIFICADO (CRITICO)** | Misma correccion de columnas en comparacion de modelos |
| 14.1 | **NUEVO** | Analisis de ablacion (4 escenarios) |
| 14.2 | **NUEVO** | Validacion temporal |
| 14.3 | **NUEVO** | Comparacion antes/despues con narrativa |
| 15 | **NUEVO** | Arquitectura tecnica del sistema NBO |
| 16 | **NUEVO** | Esquema conceptual de despliegue |
| 17 | **NUEVO** | Conclusiones finales, limitaciones, recomendaciones |

## Graficas Nuevas (carpeta `resultados/graficas23Febrero/`)

1. `fuga_sin_tarifa_vs_target.png` - Evidencia de fuga en SIN TARIFA
2. `sospecha_tipo_cliente_vs_target.png` - Analisis de MSH_TIPO_CLIENTE
3. `correlacion_variables_target.png` - Correlacion de todas las variables con target
4. `decision_variables_tabla.png` - Tabla de decisiones por variable
5. `ablacion_comparativa_metricas.png` - Comparacion de metricas por escenario
6. `ablacion_importancia_features_por_escenario.png` - Top features por escenario
7. `validacion_temporal_vs_aleatoria.png` - Temporal vs aleatorio
8. `validacion_temporal_metricas.png` - Curvas ROC temporales
9. `comparacion_antes_despues_metricas.png` - Metricas antes/despues
10. `comparacion_antes_despues_curvas_roc.png` - Curvas ROC antes/despues
11. `esquema_despliegue.png` - Timeline de despliegue

---

## Pendiente

Para generar los resultados finales, se debe ejecutar el notebook completo (Kernel -> Restart & Run All) con el archivo `Total_Mes_Act_Datos completos CORREGIDO.csv` en la raiz del proyecto.
