import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc, 
                            precision_recall_curve)
import warnings
import os
import json

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

output_dir = 'resultados'
os.makedirs(output_dir, exist_ok=True)

print('=' * 80)
print('PASO 14: COMPARACIÓN DE MODELOS')
print('=' * 80)

print('\nCargando y preparando datos...', flush=True)

archivo = 'Total_Mes_Act_Datos completos.csv'
df = pd.read_csv(
    archivo, 
    low_memory=False,
    encoding='utf-8',
    on_bad_lines='skip',
    sep=';',
    quotechar='"'
)

print(f'✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas', flush=True)

columna_tarifa = 'TARIFA_NBO'
columna_rentabilizacion = 'Rentabilizo'

df['variable_objetivo'] = np.where(
    (df[columna_tarifa].notna()) & 
    (df[columna_rentabilizacion].notna()),
    1,
    0
)

y = df['variable_objetivo'].copy()
X = df.drop(columns=['variable_objetivo', columna_tarifa, columna_rentabilizacion])

print(f'\nVariable objetivo creada:', flush=True)
print(f'  Clase 0 (No cumple): {sum(y == 0):,} registros ({sum(y == 0)/len(y)*100:.2f}%)', flush=True)
print(f'  Clase 1 (Cumple): {sum(y == 1):,} registros ({sum(y == 1)/len(y)*100:.2f}%)', flush=True)

columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()

X_processed = X[columnas_numericas].copy()

print(f'\nProcesando {len(columnas_categoricas)} columnas categóricas...', flush=True)
for i, col in enumerate(columnas_categoricas):
    if X[col].nunique() < 50:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X[col].astype(str).fillna('Missing'))
    if (i + 1) % 10 == 0:
        print(f'  Procesadas {i + 1}/{len(columnas_categoricas)} columnas...', flush=True)

X_processed = X_processed.fillna(X_processed.median())

print(f'\n✓ Datos procesados: {X_processed.shape[1]} features', flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f'✓ División train/test:', flush=True)
print(f'  Train: {X_train.shape[0]:,} muestras', flush=True)
print(f'  Test: {X_test.shape[0]:,} muestras', flush=True)

print('\n' + '=' * 80)
print('ENTRENAMIENTO DE MODELOS')
print('=' * 80)

modelos = {}
resultados_modelos = {}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\n1. Regresión Logística...', flush=True)
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

modelos['LogisticRegression'] = lr
resultados_modelos['LogisticRegression'] = {
    'y_pred': y_pred_lr,
    'y_pred_proba': y_pred_proba_lr,
    'scaler': scaler
}
print('  ✓ Regresión Logística entrenada', flush=True)

print('\n2. Random Forest...', flush=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

modelos['RandomForest'] = rf
resultados_modelos['RandomForest'] = {
    'y_pred': y_pred_rf,
    'y_pred_proba': y_pred_proba_rf,
    'scaler': None
}
print('  ✓ Random Forest entrenado', flush=True)

print('\n3. XGBoost...', flush=True)
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                    random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]

modelos['XGBoost'] = xgb
resultados_modelos['XGBoost'] = {
    'y_pred': y_pred_xgb,
    'y_pred_proba': y_pred_proba_xgb,
    'scaler': None
}
print('  ✓ XGBoost entrenado', flush=True)

print('\n✓ Todos los modelos entrenados exitosamente', flush=True)

print('\n' + '=' * 80)
print('CÁLCULO DE MÉTRICAS')
print('=' * 80)

metricas_comparacion = []

for nombre, modelo in modelos.items():
    y_pred = resultados_modelos[nombre]['y_pred']
    y_pred_proba = resultados_modelos[nombre]['y_pred_proba']
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    metricas_comparacion.append({
        'Modelo': nombre,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    })
    
    print(f'\n{nombre}:', flush=True)
    print(f'  Accuracy:  {accuracy:.4f}', flush=True)
    print(f'  Precision: {precision:.4f}', flush=True)
    print(f'  Recall:    {recall:.4f}', flush=True)
    print(f'  F1-Score:  {f1:.4f}', flush=True)
    print(f'  ROC-AUC:   {roc_auc:.4f}', flush=True)
    print(f'  PR-AUC:    {pr_auc:.4f}', flush=True)

df_metricas = pd.DataFrame(metricas_comparacion)
df_metricas.to_csv(os.path.join(output_dir, 'comparacion_modelos_metricas.csv'), index=False)
print(f'\n✓ Métricas guardadas en: comparacion_modelos_metricas.csv', flush=True)

mejor_modelo_accuracy = df_metricas.loc[df_metricas['Accuracy'].idxmax(), 'Modelo']
mejor_modelo_f1 = df_metricas.loc[df_metricas['F1-Score'].idxmax(), 'Modelo']
mejor_modelo_roc = df_metricas.loc[df_metricas['ROC-AUC'].idxmax(), 'Modelo']

print('\n' + '=' * 80)
print('MEJOR MODELO POR MÉTRICA')
print('=' * 80)
print(f'  Mejor Accuracy:  {mejor_modelo_accuracy} ({df_metricas.loc[df_metricas["Accuracy"].idxmax(), "Accuracy"]:.4f})', flush=True)
print(f'  Mejor F1-Score:  {mejor_modelo_f1} ({df_metricas.loc[df_metricas["F1-Score"].idxmax(), "F1-Score"]:.4f})', flush=True)
print(f'  Mejor ROC-AUC:   {mejor_modelo_roc} ({df_metricas.loc[df_metricas["ROC-AUC"].idxmax(), "ROC-AUC"]:.4f})', flush=True)

print('\n' + '=' * 80)
print('GENERANDO VISUALIZACIONES COMPARATIVAS')
print('=' * 80)

fig = plt.figure(figsize=(20, 14))

ax1 = plt.subplot(2, 3, 1)
metricas_plot = df_metricas.set_index('Modelo')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
metricas_plot.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Comparación de Métricas Básicas', fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Score', fontsize=10, fontweight='bold')
ax1.set_xlabel('Modelo', fontsize=10, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
auc_plot = df_metricas.set_index('Modelo')[['ROC-AUC', 'PR-AUC']]
auc_plot.plot(kind='bar', ax=ax2, width=0.8, color=['steelblue', 'coral'])
ax2.set_title('Comparación ROC-AUC y PR-AUC', fontsize=12, fontweight='bold', pad=15)
ax2.set_ylabel('AUC Score', fontsize=10, fontweight='bold')
ax2.set_xlabel('Modelo', fontsize=10, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
for nombre in modelos.keys():
    y_pred_proba = resultados_modelos[nombre]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax3.plot(fpr, tpr, lw=2, label=f'{nombre} (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax3.set_xlabel('Tasa de Falsos Positivos', fontsize=10, fontweight='bold')
ax3.set_ylabel('Tasa de Verdaderos Positivos', fontsize=10, fontweight='bold')
ax3.set_title('Curvas ROC Comparativas', fontsize=12, fontweight='bold', pad=15)
ax3.legend(loc="lower right", fontsize=9)
ax3.grid(alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
for nombre in modelos.keys():
    y_pred_proba = resultados_modelos[nombre]['y_pred_proba']
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    ax4.plot(recall_curve, precision_curve, lw=2, label=f'{nombre} (AUC = {pr_auc:.4f})')
baseline = len(y_test[y_test==1]) / len(y_test)
ax4.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.4f})')
ax4.set_xlabel('Recall', fontsize=10, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=10, fontweight='bold')
ax4.set_title('Curvas Precision-Recall Comparativas', fontsize=12, fontweight='bold', pad=15)
ax4.legend(loc="lower left", fontsize=9)
ax4.grid(alpha=0.3)

for idx, nombre in enumerate(modelos.keys(), 1):
    ax = plt.subplot(2, 3, 4 + idx)
    y_pred = resultados_modelos[nombre]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Clase 0', 'Clase 1'], 
                yticklabels=['Clase 0', 'Clase 1'])
    ax.set_title(f'Matriz Confusión - {nombre}', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Predicción', fontsize=9, fontweight='bold')
    ax.set_ylabel('Real', fontsize=9, fontweight='bold')

plt.tight_layout()
ruta_comparacion = os.path.join(output_dir, 'comparacion_modelos_visualizacion.png')
plt.savefig(ruta_comparacion, dpi=300, bbox_inches='tight')
print(f'\n✓ Visualización comparativa guardada en: {ruta_comparacion}', flush=True)
plt.close()

print('\n' + '=' * 80)
print('VALIDACIÓN CRUZADA (5 FOLDS)')
print('=' * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados_cv = {}

print('\nEjecutando validación cruzada...', flush=True)

scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
resultados_cv['LogisticRegression'] = {
    'scores': scores_lr,
    'mean': scores_lr.mean(),
    'std': scores_lr.std()
}
print(f'  Regresión Logística: {scores_lr.mean():.4f} (+/- {scores_lr.std() * 2:.4f})', flush=True)

scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
resultados_cv['RandomForest'] = {
    'scores': scores_rf,
    'mean': scores_rf.mean(),
    'std': scores_rf.std()
}
print(f'  Random Forest: {scores_rf.mean():.4f} (+/- {scores_rf.std() * 2:.4f})', flush=True)

scores_xgb = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
resultados_cv['XGBoost'] = {
    'scores': scores_xgb,
    'mean': scores_xgb.mean(),
    'std': scores_xgb.std()
}
print(f'  XGBoost: {scores_xgb.mean():.4f} (+/- {scores_xgb.std() * 2:.4f})', flush=True)

cv_comparacion = pd.DataFrame({
    'Modelo': list(resultados_cv.keys()),
    'CV_Mean': [resultados_cv[m]['mean'] for m in resultados_cv.keys()],
    'CV_Std': [resultados_cv[m]['std'] for m in resultados_cv.keys()],
    'CV_Min': [resultados_cv[m]['scores'].min() for m in resultados_cv.keys()],
    'CV_Max': [resultados_cv[m]['scores'].max() for m in resultados_cv.keys()]
})

cv_comparacion.to_csv(os.path.join(output_dir, 'comparacion_modelos_cv.csv'), index=False)
print(f'\n✓ Resultados CV guardados en: comparacion_modelos_cv.csv', flush=True)

mejor_cv = cv_comparacion.loc[cv_comparacion['CV_Mean'].idxmax(), 'Modelo']
mejor_cv_score = cv_comparacion.loc[cv_comparacion['CV_Mean'].idxmax(), 'CV_Mean']

print('\n' + '=' * 80)
print('CONCLUSIÓN - MEJOR MODELO')
print('=' * 80)
print(f'\n✓ Mejor modelo según Validación Cruzada: {mejor_cv}', flush=True)
print(f'  Accuracy promedio CV: {mejor_cv_score:.4f}', flush=True)
print(f'  Desviación estándar: {cv_comparacion.loc[cv_comparacion["CV_Mean"].idxmax(), "CV_Std"]:.4f}', flush=True)

conclusion = {
    'mejor_modelo_cv': mejor_cv,
    'mejor_score_cv': float(mejor_cv_score),
    'metricas_test': df_metricas.to_dict('records'),
    'resultados_cv': {
        k: {
            'mean': float(v['mean']),
            'std': float(v['std']),
            'scores': [float(x) for x in v['scores']]
        }
        for k, v in resultados_cv.items()
    }
}

ruta_conclusion = os.path.join(output_dir, 'conclusion_paso_14.json')
with open(ruta_conclusion, 'w', encoding='utf-8') as f:
    json.dump(conclusion, f, indent=2)

print(f'\n✓ Conclusión guardada en: {ruta_conclusion}', flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
cv_comparacion.set_index('Modelo')['CV_Mean'].plot(kind='bar', ax=ax1, color=['steelblue', 'coral', 'green'])
ax1.set_title('Accuracy Promedio - Validación Cruzada (5 folds)', fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
ax1.set_xlabel('Modelo', fontsize=10, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(cv_comparacion['CV_Mean'].values):
    ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2 = axes[1]
x_pos = np.arange(len(cv_comparacion))
ax2.bar(x_pos, cv_comparacion['CV_Mean'], yerr=cv_comparacion['CV_Std'], 
        color=['steelblue', 'coral', 'green'], alpha=0.7, capsize=5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(cv_comparacion['Modelo'], rotation=45, ha='right')
ax2.set_title('Accuracy CV con Intervalos de Confianza', fontsize=12, fontweight='bold', pad=15)
ax2.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
ax2.set_xlabel('Modelo', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
ruta_cv_viz = os.path.join(output_dir, 'comparacion_modelos_cv_visualizacion.png')
plt.savefig(ruta_cv_viz, dpi=300, bbox_inches='tight')
print(f'✓ Visualización CV guardada en: {ruta_cv_viz}', flush=True)
plt.close()

print('\n✓ Paso 14 completado exitosamente', flush=True)

