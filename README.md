# Analisis de Datos NBO - Machine Learning

Este proyecto contiene el analisis de datos y modelado predictivo para el sistema NBO (Next Best Offer). Incluye deteccion y correccion de fuga de datos (data leakage), validacion critica de modelos, y arquitectura tecnica de despliegue.

## Archivos del Proyecto

- `analisis_datos.ipynb` - Notebook principal con analisis corregido (22 pasos)
- `convertir_excel_a_csv.py` - Script para convertir archivos Excel a CSV
- `requirements.txt` - Dependencias del proyecto
- `RESPUESTA_COMENTARIOS_TUTORA.md` - Documentacion de correcciones aplicadas
- `CONCLUSIONES_ANALISIS.md` - Conclusiones del analisis

## Requisitos Previos

### 1. Python 3.7+

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Las dependencias incluyen: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, scipy, openpyxl, jupyter, ipykernel.

### 3. Dataset (obligatorio)

Colocar el archivo CSV en la raiz del proyecto:

```
Total_Mes_Act_Datos completos CORREGIDO.csv
```

Este archivo **no esta incluido en el repositorio** (esta en `.gitignore` por su tamano). Debe obtenerse del equipo de datos o de la fuente original.

El archivo debe contener las siguientes columnas (entre otras):
- `TARIFA_NBO`, `Rentabilizo` (componentes del target)
- `Msh Cuenta`, `MSH_CEDULA`, `MSH_LOGIN_ID` (identificadores)
- `SIN TARIFA`, `MSH_TIPO_CLIENTE`, `MOVIMIENTO_RENTA` (variables analizadas por fuga)
- `Msh Fecha (copia)` (usada para validacion temporal)

## Como Ejecutar el Notebook Corregido

### Opcion A: Desde Jupyter (recomendado)

```bash
jupyter notebook
```

1. Abrir `analisis_datos.ipynb`
2. Ejecutar **Kernel -> Restart & Run All**
3. Esperar a que todas las celdas se ejecuten secuencialmente

### Opcion B: Desde linea de comandos

```bash
jupyter nbconvert --to notebook --execute analisis_datos.ipynb --output analisis_datos_ejecutado.ipynb
```

### Opcion C: Desde VS Code

1. Abrir `analisis_datos.ipynb` en VS Code
2. Seleccionar el kernel de Python con las dependencias instaladas
3. Ejecutar todas las celdas (boton "Run All" o `Ctrl+Shift+Enter`)

## Estructura del Notebook (22 Pasos)

| Paso | Descripcion |
|---|---|
| 1 | Importacion de librerias y creacion de carpetas |
| 2 | Carga de datos desde CSV |
| 3 | Conversion a JSON y conteo por cuenta |
| 4 | Analisis Exploratorio de Datos (EDA) |
| **4.1** | **EDA con decisiones concretas por variable** |
| 5 | Identificacion y filtrado de datos |
| **5.1** | **Documentacion de variables eliminadas** |
| **6** | **Preparacion de datos (corregido: sin fuga de datos)** |
| 7 | Entrenamiento de modelos |
| 8 | Grid Search con muestreo estratificado |
| 9 | Resumen de resultados de Grid Search |
| 10 | Comparacion y conclusion |
| 11 | Analisis de importancia de variables |
| 12 | Visualizacion de arboles de decision |
| 13 | Visualizaciones EDA individuales |
| **14** | **Comparacion de modelos (corregido: sin fuga de datos)** |
| **14.1** | **Analisis de ablacion (4 escenarios)** |
| **14.2** | **Validacion temporal** |
| **14.3** | **Comparacion antes/despues de correcciones** |
| **15** | **Arquitectura tecnica del sistema NBO** |
| **16** | **Esquema conceptual de despliegue** |
| **17** | **Conclusiones finales, limitaciones y recomendaciones** |

Los pasos en **negrita** son nuevos o modificados en la correccion del 23 de febrero de 2026.

## Archivos de Salida

Todos los resultados se guardan en `resultados/`:

### Resultados generales
- `datos_completos.json` - Conversion del CSV a JSON
- `conteo_por_cuenta.csv` - Conteo por campo cuenta
- `importancia_caracteristicas.csv` - Importancia de features
- `mapeo_features_nombres_reales.json` - Mapeo Feature_X a nombres reales

### Resultados Paso 14 (`resultados/paso14/`)
- `comparacion_modelos_metricas.csv` - Metricas de los 3 modelos
- `conclusion_paso14.json` - Conclusion con mejor modelo
- `resultados_validacion_cruzada_paso14.json` - Resultados CV

### Graficas de correcciones (`resultados/graficas23Febrero/`)
| Archivo | Descripcion |
|---|---|
| `fuga_sin_tarifa_vs_target.png` | Evidencia de fuga en SIN TARIFA |
| `sospecha_tipo_cliente_vs_target.png` | Analisis de MSH_TIPO_CLIENTE |
| `correlacion_variables_target.png` | Correlacion de variables con target |
| `decision_variables_tabla.png` | Tabla de decisiones por variable |
| `ablacion_comparativa_metricas.png` | Metricas por escenario de ablacion |
| `ablacion_importancia_features_por_escenario.png` | Top features por escenario |
| `validacion_temporal_vs_aleatoria.png` | Temporal vs aleatorio |
| `validacion_temporal_metricas.png` | Curvas ROC temporales |
| `comparacion_antes_despues_metricas.png` | Metricas antes/despues |
| `comparacion_antes_despues_curvas_roc.png` | Curvas ROC antes/despues |
| `esquema_despliegue.png` | Timeline de despliegue |
| `documentacion_variables_eliminadas.json` | Justificacion de exclusiones |
| `resultados_ablacion.json` | Resultados numericos de ablacion |

## Verificacion Post-Ejecucion

Despues de ejecutar el notebook, verificar:

1. La carpeta `resultados/graficas23Febrero/` contiene las 11+ graficas
2. Las metricas del modelo corregido son **menores** que las originales (esperado: AUC ~0.75-0.92 en vez de 0.999)
3. Los 4 escenarios de ablacion (A, B, C, D) se ejecutan sin error
4. La validacion temporal se ejecuta o documenta su limitacion si la fecha no es parseable
5. Los Pasos 15-17 muestran contenido completo de arquitectura y despliegue

## Notas

- El notebook detecta automaticamente las columnas `TARIFA_NBO` y `Rentabilizo`
- Si el CSV es muy grande (>200MB), la conversion a JSON se limita a 100,000 registros
- El Grid Search y la ablacion pueden tardar varios minutos dependiendo del tamano de los datos
- Si `Msh Fecha (copia)` no puede parsearse como fecha, el Paso 14.2 documenta la limitacion en lugar de fallar
