"""
Script para análisis de datos con modelos de machine learning.
Procesa datos del CSV, realiza EDA y compara modelos de clasificación.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
import json
from datetime import datetime
import os
warnings.filterwarnings('ignore')

def cargar_datos(archivo):
    """
    Carga el archivo CSV de forma eficiente.
    
    Args:
        archivo: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    print("Cargando datos...")
    try:
        df = pd.read_csv(
            archivo, 
            low_memory=False,
            encoding='utf-8',
            on_bad_lines='skip',
            sep=',',
            quotechar='"'
        )
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        print("Intentando con diferentes parámetros...")
        try:
            df = pd.read_csv(
                archivo,
                low_memory=False,
                encoding='latin-1',
                on_bad_lines='skip',
                sep=';',
                quotechar='"'
            )
            print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e2:
            print(f"Error al cargar datos con parámetros alternativos: {e2}")
            return None

def convertir_csv_a_json(df, archivo_json, output_dir='resultados', max_rows=None):
    """
    Convierte el DataFrame a formato JSON.
    
    Args:
        df: DataFrame a convertir
        archivo_json: Nombre del archivo JSON de salida
        output_dir: Directorio de salida
        max_rows: Número máximo de filas a convertir (None para todas)
    """
    os.makedirs(output_dir, exist_ok=True)
    ruta_json = os.path.join(output_dir, archivo_json)
    
    print(f"\nConvirtiendo datos a JSON...")
    
    df_convertir = df.head(max_rows) if max_rows else df
    
    try:
        df_convertir.to_json(
            ruta_json,
            orient='records',
            date_format='iso',
            indent=2,
            force_ascii=False
        )
        print(f"Archivo JSON guardado en '{ruta_json}'")
        print(f"Total de registros convertidos: {len(df_convertir)}")
        return ruta_json
    except Exception as e:
        print(f"Error al convertir a JSON: {e}")
        print("Intentando con formato alternativo...")
        try:
            df_convertir.to_json(
                ruta_json,
                orient='records',
                date_format='iso',
                force_ascii=False
            )
            print(f"Archivo JSON guardado en '{ruta_json}' (sin indentación)")
            return ruta_json
        except Exception as e2:
            print(f"Error al convertir a JSON con formato alternativo: {e2}")
            return None

def conteo_por_cuenta(df, output_dir='resultados'):
    """
    Realiza un conteo por el campo 'cuenta' y guarda los resultados.
    
    Args:
        df: DataFrame con los datos
        output_dir: Directorio de salida
        
    Returns:
        DataFrame con el conteo por cuenta
    """
    print("\n" + "="*80)
    print("CONTEO POR CAMPO 'CUENTA'")
    print("="*80)
    
    columna_cuenta = None
    
    for col in df.columns:
        if 'cuenta' in col.lower():
            columna_cuenta = col
            break
    
    if columna_cuenta is None:
        print("No se encontró columna 'cuenta'. Buscando columnas similares...")
        posibles = [col for col in df.columns if 'cuent' in col.lower() or 'account' in col.lower()]
        if posibles:
            columna_cuenta = posibles[0]
            print(f"Usando columna: {columna_cuenta}")
        else:
            print("No se encontró columna relacionada con 'cuenta'.")
            print("Columnas disponibles (primeras 20):")
            print(df.columns[:20].tolist())
            return None
    
    print(f"\nColumna seleccionada: {columna_cuenta}")
    
    conteo = df[columna_cuenta].value_counts().reset_index()
    conteo.columns = ['Cuenta', 'Frecuencia']
    conteo = conteo.sort_values('Frecuencia', ascending=False)
    
    print(f"\nTotal de cuentas únicas: {len(conteo)}")
    print(f"\nTop 10 cuentas más frecuentes:")
    print(conteo.head(10).to_string(index=False))
    
    os.makedirs(output_dir, exist_ok=True)
    
    archivo_csv = os.path.join(output_dir, 'conteo_por_cuenta.csv')
    conteo.to_csv(archivo_csv, index=False, encoding='utf-8')
    print(f"\nConteo guardado en '{archivo_csv}'")
    
    archivo_json = os.path.join(output_dir, 'conteo_por_cuenta.json')
    conteo.to_json(archivo_json, orient='records', indent=2, force_ascii=False)
    print(f"Conteo en JSON guardado en '{archivo_json}'")
    
    archivo_txt = os.path.join(output_dir, 'conteo_por_cuenta.txt')
    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONTEO POR CAMPO 'CUENTA'\n")
        f.write("="*80 + "\n\n")
        f.write(f"Columna analizada: {columna_cuenta}\n")
        f.write(f"Total de cuentas únicas: {len(conteo)}\n")
        f.write(f"Total de registros: {conteo['Frecuencia'].sum()}\n\n")
        f.write("Distribución completa:\n")
        f.write("-" * 80 + "\n")
        f.write(conteo.to_string(index=False))
        f.write("\n\n")
        f.write("Estadísticas:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Frecuencia mínima: {conteo['Frecuencia'].min()}\n")
        f.write(f"Frecuencia máxima: {conteo['Frecuencia'].max()}\n")
        f.write(f"Frecuencia promedio: {conteo['Frecuencia'].mean():.2f}\n")
        f.write(f"Frecuencia mediana: {conteo['Frecuencia'].median():.2f}\n")
    
    print(f"Resumen de conteo guardado en '{archivo_txt}'")
    
    return conteo

def exploracion_datos(df):
    """
    Realiza análisis exploratorio de datos (EDA).
    
    Args:
        df: DataFrame con los datos
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*80)
    
    print("\n1. Información general del dataset:")
    print(df.info())
    
    print("\n2. Primeras filas:")
    print(df.head())
    
    print("\n3. Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n4. Valores faltantes por columna:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores_Faltantes': missing,
        'Porcentaje': missing_pct
    })
    missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
    print(missing_df)
    
    print("\n5. Tipos de datos:")
    print(df.dtypes.value_counts())
    
    print("\n6. Columnas del dataset:")
    print(df.columns.tolist())
    
    return missing_df

def filtrar_datos(df, columna_tarifa, columna_target):
    """
    Filtra datos donde la columna de tarifa tiene información.
    
    Args:
        df: DataFrame con los datos
        columna_tarifa: Nombre de la columna de tarifa NBO
        columna_target: Nombre de la columna objetivo (Rentabilizacion)
        
    Returns:
        DataFrame filtrado
    """
    print("\n" + "="*80)
    print("FILTRADO DE DATOS")
    print("="*80)
    
    print(f"\nFilas originales: {len(df)}")
    
    if columna_tarifa not in df.columns:
        print(f"Error: Columna '{columna_tarifa}' no encontrada.")
        print("Columnas disponibles:")
        print([col for col in df.columns if 'tarifa' in col.lower() or 'nbo' in col.lower()])
        return None
    
    if columna_target not in df.columns:
        print(f"Error: Columna '{columna_target}' no encontrada.")
        print("Columnas disponibles:")
        print([col for col in df.columns if 'rentabilizacion' in col.lower() or 'rent' in col.lower()])
        return None
    
    df_filtrado = df[df[columna_tarifa].notna()].copy()
    print(f"Filas después de filtrar por {columna_tarifa}: {len(df_filtrado)}")
    
    df_filtrado = df_filtrado[df_filtrado[columna_target].notna()].copy()
    print(f"Filas después de filtrar por {columna_target}: {len(df_filtrado)}")
    
    print(f"\nDistribución de la variable objetivo ({columna_target}):")
    print(df_filtrado[columna_target].value_counts())
    
    return df_filtrado

def preparar_datos(df, columna_target):
    """
    Prepara los datos para el modelado.
    
    Args:
        df: DataFrame filtrado
        columna_target: Nombre de la columna objetivo
        
    Returns:
        X: Features
        y: Target variable
    """
    print("\n" + "="*80)
    print("PREPARACIÓN DE DATOS")
    print("="*80)
    
    y = df[columna_target].copy()
    
    X = df.drop(columns=[columna_target])
    
    print(f"\nFeatures originales: {X.shape[1]}")
    
    columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Columnas numéricas: {len(columnas_numericas)}")
    print(f"Columnas categóricas: {len(columnas_categoricas)}")
    
    X_processed = X[columnas_numericas].copy()
    
    for col in columnas_categoricas:
        if X[col].nunique() < 50:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col].astype(str).fillna('Missing'))
        else:
            print(f"Omitiendo columna categórica '{col}' con {X[col].nunique()} valores únicos")
    
    X_processed = X_processed.fillna(X_processed.median())
    
    if isinstance(y.dtype, object) or y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        print(f"\nVariable objetivo codificada. Clases: {np.unique(y)}")
    else:
        y = y.astype(int)
        print(f"\nVariable objetivo. Clases: {np.unique(y)}")
    
    print(f"\nShape final: X={X_processed.shape}, y={y.shape}")
    
    return X_processed, y

def entrenar_modelos(X, y):
    """
    Entrena y compara diferentes modelos de machine learning.
    
    Args:
        X: Features
        y: Target variable
        
    Returns:
        Diccionario con los modelos entrenados y sus resultados
    """
    print("\n" + "="*80)
    print("ENTRENAMIENTO DE MODELOS")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    resultados = {}
    
    print("\n1. Regresión Logística")
    print("-" * 80)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    resultados['LogisticRegression'] = {
        'modelo': lr,
        'scaler': scaler,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]) if len(np.unique(y)) == 2 else None,
        'y_pred': y_pred_lr,
        'y_test': y_test
    }
    print(f"Accuracy: {resultados['LogisticRegression']['accuracy']:.4f}")
    
    print("\n2. Random Forest")
    print("-" * 80)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    resultados['RandomForest'] = {
        'modelo': rf,
        'scaler': None,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]) if len(np.unique(y)) == 2 else None,
        'y_pred': y_pred_rf,
        'y_test': y_test
    }
    print(f"Accuracy: {resultados['RandomForest']['accuracy']:.4f}")
    
    print("\n3. XGBoost")
    print("-" * 80)
    xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    resultados['XGBoost'] = {
        'modelo': xgb,
        'scaler': None,
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]) if len(np.unique(y)) == 2 else None,
        'y_pred': y_pred_xgb,
        'y_test': y_test
    }
    print(f"Accuracy: {resultados['XGBoost']['accuracy']:.4f}")
    
    return resultados, X_train, X_test, y_train, y_test

def validacion_cruzada(X, y):
    """
    Realiza validación cruzada para todos los modelos.
    
    Args:
        X: Features
        y: Target variable
        
    Returns:
        Diccionario con resultados de validación cruzada
    """
    print("\n" + "="*80)
    print("VALIDACIÓN CRUZADA")
    print("="*80)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    resultados_cv = {}
    
    print("\n1. Regresión Logística - CV")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    scores_lr = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')
    resultados_cv['LogisticRegression'] = {
        'scores': scores_lr,
        'mean': scores_lr.mean(),
        'std': scores_lr.std()
    }
    print(f"Accuracy: {scores_lr.mean():.4f} (+/- {scores_lr.std() * 2:.4f})")
    
    print("\n2. Random Forest - CV")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    resultados_cv['RandomForest'] = {
        'scores': scores_rf,
        'mean': scores_rf.mean(),
        'std': scores_rf.std()
    }
    print(f"Accuracy: {scores_rf.mean():.4f} (+/- {scores_rf.std() * 2:.4f})")
    
    print("\n3. XGBoost - CV")
    xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    scores_xgb = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy')
    resultados_cv['XGBoost'] = {
        'scores': scores_xgb,
        'mean': scores_xgb.mean(),
        'std': scores_xgb.std()
    }
    print(f"Accuracy: {scores_xgb.mean():.4f} (+/- {scores_xgb.std() * 2:.4f})")
    
    return resultados_cv

def grid_search_arboles(X, y):
    """
    Realiza grid search para los modelos de árboles de decisión.
    
    Args:
        X: Features
        y: Target variable
        
    Returns:
        Diccionario con los mejores modelos y parámetros
    """
    print("\n" + "="*80)
    print("GRID SEARCH PARA ÁRBOLES DE DECISIÓN")
    print("="*80)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n1. Grid Search - Random Forest")
    print("-" * 80)
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_rf = GridSearchCV(
        rf_base, param_grid_rf, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    grid_rf.fit(X, y)
    
    print(f"Mejores parámetros RF: {grid_rf.best_params_}")
    print(f"Mejor score RF: {grid_rf.best_score_:.4f}")
    
    print("\n2. Grid Search - XGBoost")
    print("-" * 80)
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb_base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    grid_xgb = GridSearchCV(
        xgb_base, param_grid_xgb, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid_xgb.fit(X, y)
    
    print(f"Mejores parámetros XGBoost: {grid_xgb.best_params_}")
    print(f"Mejor score XGBoost: {grid_xgb.best_score_:.4f}")
    
    resultados_grid = {
        'RandomForest': {
            'best_model': grid_rf.best_estimator_,
            'best_params': grid_rf.best_params_,
            'best_score': grid_rf.best_score_
        },
        'XGBoost': {
            'best_model': grid_xgb.best_estimator_,
            'best_params': grid_xgb.best_params_,
            'best_score': grid_xgb.best_score_
        }
    }
    
    return resultados_grid

def comparar_arboles(resultados_cv, resultados_grid):
    """
    Compara los modelos de árboles de decisión y determina el mejor.
    
    Args:
        resultados_cv: Resultados de validación cruzada
        resultados_grid: Resultados de grid search
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE ÁRBOLES DE DECISIÓN")
    print("="*80)
    
    print("\nResultados de Validación Cruzada:")
    print("-" * 80)
    print(f"Random Forest - Accuracy: {resultados_cv['RandomForest']['mean']:.4f} (+/- {resultados_cv['RandomForest']['std'] * 2:.4f})")
    print(f"XGBoost - Accuracy: {resultados_cv['XGBoost']['mean']:.4f} (+/- {resultados_cv['XGBoost']['std'] * 2:.4f})")
    
    print("\nResultados de Grid Search:")
    print("-" * 80)
    print(f"Random Forest - Mejor Score: {resultados_grid['RandomForest']['best_score']:.4f}")
    print(f"XGBoost - Mejor Score: {resultados_grid['XGBoost']['best_score']:.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSIÓN")
    print("="*80)
    
    mejor_cv = 'RandomForest' if resultados_cv['RandomForest']['mean'] > resultados_cv['XGBoost']['mean'] else 'XGBoost'
    mejor_grid = 'RandomForest' if resultados_grid['RandomForest']['best_score'] > resultados_grid['XGBoost']['best_score'] else 'XGBoost'
    
    print(f"\nMejor modelo según Validación Cruzada: {mejor_cv}")
    print(f"Mejor modelo según Grid Search: {mejor_grid}")
    
    if mejor_cv == mejor_grid:
        print(f"\n✓ El mejor modelo de árboles de decisión es: {mejor_cv}")
    else:
        print(f"\n⚠ Hay discrepancia entre métodos. Se recomienda evaluar ambos modelos.")
        print(f"  - Validación Cruzada favorece: {mejor_cv}")
        print(f"  - Grid Search favorece: {mejor_grid}")

def guardar_resumen_eda(df, missing_df, columna_tarifa, columna_target, output_dir='resultados'):
    """
    Guarda un resumen del EDA en un archivo de texto.
    
    Args:
        df: DataFrame con los datos
        missing_df: DataFrame con valores faltantes
        columna_tarifa: Nombre de la columna de tarifa
        columna_target: Nombre de la columna objetivo
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    archivo = os.path.join(output_dir, 'resumen_eda.txt')
    
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DEL ANÁLISIS EXPLORATORIO DE DATOS (EDA)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas\n\n")
        
        f.write("Columnas seleccionadas:\n")
        f.write(f"  - Tarifa NBO: {columna_tarifa}\n")
        f.write(f"  - Rentabilizacion: {columna_target}\n\n")
        
        f.write("Distribución de la variable objetivo:\n")
        if columna_target in df.columns:
            f.write(str(df[columna_target].value_counts()) + "\n\n")
        
        f.write("Valores faltantes:\n")
        f.write(missing_df.to_string() + "\n\n")
        
        f.write("Estadísticas descriptivas:\n")
        f.write(df.describe().to_string() + "\n\n")
        
        f.write("Tipos de datos:\n")
        f.write(str(df.dtypes.value_counts()) + "\n")
    
    print(f"Resumen EDA guardado en '{archivo}'")

def guardar_resultados_modelos(resultados, output_dir='resultados'):
    """
    Guarda los resultados de los modelos en archivos CSV y texto.
    
    Args:
        resultados: Diccionario con resultados de los modelos
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    resumen_modelos = []
    for nombre, resultado in resultados.items():
        resumen_modelos.append({
            'Modelo': nombre,
            'Accuracy': resultado['accuracy'],
            'ROC_AUC': resultado['roc_auc'] if resultado['roc_auc'] is not None else 'N/A'
        })
        
        reporte = classification_report(
            resultado['y_test'], 
            resultado['y_pred'],
            output_dict=True
        )
        
        archivo_reporte = os.path.join(output_dir, f'reporte_{nombre.lower()}.txt')
        with open(archivo_reporte, 'w', encoding='utf-8') as f:
            f.write(f"Reporte de Clasificación - {nombre}\n")
            f.write("="*80 + "\n\n")
            f.write(classification_report(resultado['y_test'], resultado['y_pred']))
            f.write("\n\nMatriz de Confusión:\n")
            f.write(str(confusion_matrix(resultado['y_test'], resultado['y_pred'])))
        
        print(f"Reporte de {nombre} guardado en '{archivo_reporte}'")
    
    df_resultados = pd.DataFrame(resumen_modelos)
    archivo_csv = os.path.join(output_dir, 'resultados_modelos.csv')
    df_resultados.to_csv(archivo_csv, index=False)
    print(f"Resultados de modelos guardados en '{archivo_csv}'")

def guardar_validacion_cruzada(resultados_cv, output_dir='resultados'):
    """
    Guarda los resultados de validación cruzada.
    
    Args:
        resultados_cv: Diccionario con resultados de CV
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    resumen_cv = []
    for nombre, resultado in resultados_cv.items():
        resumen_cv.append({
            'Modelo': nombre,
            'Mean_Accuracy': resultado['mean'],
            'Std_Accuracy': resultado['std'],
            'Min_Accuracy': resultado['scores'].min(),
            'Max_Accuracy': resultado['scores'].max()
        })
    
    df_cv = pd.DataFrame(resumen_cv)
    archivo_csv = os.path.join(output_dir, 'validacion_cruzada.csv')
    df_cv.to_csv(archivo_csv, index=False)
    
    archivo_txt = os.path.join(output_dir, 'validacion_cruzada.txt')
    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("Resultados de Validación Cruzada (5 folds)\n")
        f.write("="*80 + "\n\n")
        for nombre, resultado in resultados_cv.items():
            f.write(f"{nombre}:\n")
            f.write(f"  Accuracy: {resultado['mean']:.4f} (+/- {resultado['std'] * 2:.4f})\n")
            f.write(f"  Scores por fold: {resultado['scores']}\n\n")
    
    print(f"Resultados de validación cruzada guardados en '{archivo_csv}' y '{archivo_txt}'")

def guardar_grid_search(resultados_grid, output_dir='resultados'):
    """
    Guarda los resultados del grid search.
    
    Args:
        resultados_grid: Diccionario con resultados del grid search
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    archivo_txt = os.path.join(output_dir, 'grid_search_resultados.txt')
    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("Resultados de Grid Search\n")
        f.write("="*80 + "\n\n")
        
        for nombre, resultado in resultados_grid.items():
            f.write(f"{nombre}:\n")
            f.write(f"  Mejor Score: {resultado['best_score']:.4f}\n")
            f.write(f"  Mejores Parámetros:\n")
            for param, value in resultado['best_params'].items():
                f.write(f"    {param}: {value}\n")
            f.write("\n")
    
    archivo_json = os.path.join(output_dir, 'grid_search_parametros.json')
    with open(archivo_json, 'w', encoding='utf-8') as f:
        json_result = {}
        for nombre, resultado in resultados_grid.items():
            json_result[nombre] = {
                'best_score': float(resultado['best_score']),
                'best_params': {k: str(v) for k, v in resultado['best_params'].items()}
            }
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados de grid search guardados en '{archivo_txt}' y '{archivo_json}'")

def guardar_resumen_final(resultados_cv, resultados_grid, output_dir='resultados'):
    """
    Guarda un resumen final con la comparación de modelos.
    
    Args:
        resultados_cv: Resultados de validación cruzada
        resultados_grid: Resultados de grid search
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mejor_cv = 'RandomForest' if resultados_cv['RandomForest']['mean'] > resultados_cv['XGBoost']['mean'] else 'XGBoost'
    mejor_grid = 'RandomForest' if resultados_grid['RandomForest']['best_score'] > resultados_grid['XGBoost']['best_score'] else 'XGBoost'
    
    archivo = os.path.join(output_dir, 'resumen_final.txt')
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN FINAL DEL ANÁLISIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("COMPARACIÓN DE ÁRBOLES DE DECISIÓN\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Resultados de Validación Cruzada:\n")
        f.write(f"  Random Forest: {resultados_cv['RandomForest']['mean']:.4f} (+/- {resultados_cv['RandomForest']['std'] * 2:.4f})\n")
        f.write(f"  XGBoost: {resultados_cv['XGBoost']['mean']:.4f} (+/- {resultados_cv['XGBoost']['std'] * 2:.4f})\n\n")
        
        f.write("Resultados de Grid Search:\n")
        f.write(f"  Random Forest: {resultados_grid['RandomForest']['best_score']:.4f}\n")
        f.write(f"  XGBoost: {resultados_grid['XGBoost']['best_score']:.4f}\n\n")
        
        f.write("CONCLUSIÓN:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mejor modelo según Validación Cruzada: {mejor_cv}\n")
        f.write(f"Mejor modelo según Grid Search: {mejor_grid}\n\n")
        
        if mejor_cv == mejor_grid:
            f.write(f"✓ El mejor modelo de árboles de decisión es: {mejor_cv}\n")
        else:
            f.write("⚠ Hay discrepancia entre métodos. Se recomienda evaluar ambos modelos.\n")
            f.write(f"  - Validación Cruzada favorece: {mejor_cv}\n")
            f.write(f"  - Grid Search favorece: {mejor_grid}\n")
    
    print(f"Resumen final guardado en '{archivo}'")

def visualizaciones_eda(df, columna_target, output_dir='resultados'):
    """
    Genera visualizaciones para el EDA.
    
    Args:
        df: DataFrame con los datos
        columna_target: Nombre de la columna objetivo
        output_dir: Directorio de salida
    """
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.figure(figsize=(12, 6))
        
        if columna_target in df.columns:
            plt.subplot(1, 2, 1)
            df[columna_target].value_counts().plot(kind='bar')
            plt.title(f'Distribución de {columna_target}')
            plt.xlabel(columna_target)
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns[:5]
        if len(columnas_numericas) > 0:
            plt.subplot(1, 2, 2)
            df[columnas_numericas].boxplot()
            plt.title('Distribución de Variables Numéricas')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        archivo = os.path.join(output_dir, 'eda_visualizaciones.png')
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        print(f"Visualizaciones guardadas en '{archivo}'")
        plt.close()
    except Exception as e:
        print(f"Error al generar visualizaciones: {e}")

def main():
    """
    Función principal que ejecuta todo el pipeline de análisis.
    """
    archivo = "Total_Mes_Act_Datos completos.csv"
    
    output_dir = 'resultados'
    os.makedirs(output_dir, exist_ok=True)
    
    df = cargar_datos(archivo)
    if df is None:
        return
    
    missing_df = exploracion_datos(df)
    
    print("\n" + "="*80)
    print("CONVERSIÓN A JSON Y ANÁLISIS DE CUENTAS")
    print("="*80)
    
    print("\nNota: Debido al tamaño del archivo, se convertirá una muestra representativa a JSON.")
    print("Para convertir todo el archivo, modifique el parámetro max_rows en la función.")
    
    max_rows_json = min(100000, len(df))
    convertir_csv_a_json(df, 'datos_completos.json', output_dir, max_rows=max_rows_json)
    
    conteo_cuenta = conteo_por_cuenta(df, output_dir)
    
    columna_tarifa = None
    columna_target = None
    
    for col in df.columns:
        if 'tarifa' in col.lower() and 'nbo' in col.lower():
            columna_tarifa = col
            break
    
    if columna_tarifa is None:
        print("\nBuscando columna de tarifa NBO...")
        posibles = [col for col in df.columns if 'tarifa' in col.lower() or 'nbo' in col.lower()]
        if posibles:
            columna_tarifa = posibles[0]
            print(f"Usando columna: {columna_tarifa}")
        else:
            print("No se encontró columna de tarifa NBO. Mostrando primeras columnas:")
            print(df.columns[:20].tolist())
            return
    
    for col in df.columns:
        if 'rentabilizacion' in col.lower() or 'rentabiliz' in col.lower():
            columna_target = col
            break
    
    if columna_target is None:
        print("\nBuscando columna de Rentabilizacion...")
        posibles = [col for col in df.columns if 'rent' in col.lower()]
        if posibles:
            columna_target = posibles[0]
            print(f"Usando columna: {columna_target}")
        else:
            print("No se encontró columna de Rentabilizacion. Mostrando primeras columnas:")
            print(df.columns[:20].tolist())
            return
    
    print(f"\nColumnas seleccionadas:")
    print(f"  - Tarifa NBO: {columna_tarifa}")
    print(f"  - Rentabilizacion: {columna_target}")
    
    df_filtrado = filtrar_datos(df, columna_tarifa, columna_target)
    if df_filtrado is None or len(df_filtrado) == 0:
        print("No hay datos después del filtrado.")
        return
    
    visualizaciones_eda(df_filtrado, columna_target, output_dir)
    
    guardar_resumen_eda(df_filtrado, missing_df, columna_tarifa, columna_target, output_dir)
    
    X, y = preparar_datos(df_filtrado, columna_target)
    
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Error: No hay features válidas después de la preparación.")
        return
    
    resultados, X_train, X_test, y_train, y_test = entrenar_modelos(X, y)
    
    guardar_resultados_modelos(resultados, output_dir)
    
    resultados_cv = validacion_cruzada(X, y)
    
    guardar_validacion_cruzada(resultados_cv, output_dir)
    
    resultados_grid = grid_search_arboles(X, y)
    
    guardar_grid_search(resultados_grid, output_dir)
    
    comparar_arboles(resultados_cv, resultados_grid)
    
    guardar_resumen_final(resultados_cv, resultados_grid, output_dir)
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nTodos los archivos de salida se han guardado en el directorio '{output_dir}/'")
    print("\nArchivos generados:")
    print("  - datos_completos.json (conversión del CSV)")
    print("  - conteo_por_cuenta.csv")
    print("  - conteo_por_cuenta.json")
    print("  - conteo_por_cuenta.txt")
    print("  - resumen_eda.txt")
    print("  - eda_visualizaciones.png")
    print("  - resultados_modelos.csv")
    print("  - reporte_logisticregression.txt")
    print("  - reporte_randomforest.txt")
    print("  - reporte_xgboost.txt")
    print("  - validacion_cruzada.csv")
    print("  - validacion_cruzada.txt")
    print("  - grid_search_resultados.txt")
    print("  - grid_search_parametros.json")
    print("  - resumen_final.txt")

if __name__ == "__main__":
    main()

