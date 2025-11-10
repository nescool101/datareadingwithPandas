# Análisis de Datos - Machine Learning

Este proyecto contiene herramientas para analizar datos con modelos de machine learning.

## Archivos del Proyecto

- `convertir_excel_a_csv.py` - Script para convertir archivos Excel a CSV
- `analisis_datos.py` - Script completo de análisis (ejecutable desde línea de comandos)
- `analisis_datos.ipynb` - Jupyter notebook con análisis paso a paso
- `requirements.txt` - Dependencias del proyecto

## Instalación

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Paso 1: Convertir Excel a CSV

Si tienes un archivo Excel que necesitas convertir:

```bash
python convertir_excel_a_csv.py archivo.xlsx
```

O especificando el archivo de salida y la hoja:

```bash
python convertir_excel_a_csv.py archivo.xlsx archivo.csv "Hoja1"
```

### Paso 2: Análisis con Script Python

Ejecutar el script completo:

```bash
python analisis_datos.py
```

Este script:
- Carga el CSV
- Convierte a JSON (muestra de 100,000 registros)
- Realiza conteo por campo "cuenta"
- Genera análisis EDA
- Filtra datos por "tarifa NBO" y "Rentabilizacion"
- Entrena modelos (Regresión Logística, Random Forest, XGBoost)
- Realiza validación cruzada
- Ejecuta Grid Search
- Compara modelos y determina el mejor

### Paso 3: Análisis con Jupyter Notebook (Recomendado)

Para ver el progreso paso a paso:

1. Iniciar Jupyter:
```bash
jupyter notebook
```

2. Abrir `analisis_datos.ipynb`

3. Ejecutar las celdas en orden para ver el progreso del análisis

## Archivos de Salida

Todos los resultados se guardan en el directorio `resultados/`:

- `datos_completos.json` - Conversión del CSV a JSON
- `conteo_por_cuenta.csv` - Conteo por campo cuenta
- `conteo_por_cuenta.json` - Conteo en formato JSON
- `conteo_por_cuenta.txt` - Resumen del conteo
- `resumen_eda.txt` - Resumen del análisis exploratorio
- `eda_visualizaciones.png` - Visualizaciones del EDA
- `resultados_modelos.csv` - Resultados de los modelos
- `reporte_logisticregression.txt` - Reporte de Regresión Logística
- `reporte_randomforest.txt` - Reporte de Random Forest
- `reporte_xgboost.txt` - Reporte de XGBoost
- `validacion_cruzada.csv` - Resultados de validación cruzada
- `validacion_cruzada.txt` - Resumen de validación cruzada
- `grid_search_resultados.txt` - Resultados del Grid Search
- `grid_search_parametros.json` - Parámetros óptimos en JSON
- `resumen_final.txt` - Resumen final con la comparación de modelos

## Requisitos

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- openpyxl (para Excel)
- jupyter (para notebook)

## Notas

- El script detecta automáticamente las columnas "tarifa NBO" y "Rentabilizacion"
- Si el archivo CSV es muy grande (>200MB), la conversión a JSON se limita a 100,000 registros
- El Grid Search puede tardar varios minutos dependiendo del tamaño de los datos

