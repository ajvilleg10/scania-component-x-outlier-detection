# SCANIA Component X - Detección de outliers temporales multivariados

Proyecto automatizado para el TFM **Detección de outliers en series temporales multivariadas para mantenimiento predictivo en vehículos pesados**.

El proyecto está diseñado para ejecutarse principalmente en **Google Colab conectado a Google Drive**, aprovechando GPU para el entrenamiento de modelos de aprendizaje profundo. Los notebooks quedan como apoyo explicativo, de visualización y defensa; la ejecución oficial se realiza mediante `main.py`.

---

## 1. Objetivo del proyecto

Implementar un pipeline reproducible para detectar **outliers temporales multivariados** en lecturas operacionales del dataset **SCANIA Component X**, comparando tres modelos de aprendizaje profundo:

- `lstm_autoencoder`
- `cnn_lstm_autoencoder`
- `transformer_encoder_simplified`

Los modelos generan una **puntuación de atipicidad** u `outlier_score` por ventana temporal. La evaluación principal se realiza a nivel **vehículo/trayectoria**, agregando los scores de ventana y comparándolos con las etiquetas disponibles (`class_label`) cuando corresponda.

---

## 2. Principios metodológicos implementados

El pipeline respeta las decisiones metodológicas finales del TFM:

1. El dataset ya proporciona particiones oficiales de `train`, `validation` y `test`.
2. No se realiza una nueva división aleatoria del dataset.
3. No se aplica validación cruzada.
4. `train` se usa para ajustar el preprocesamiento y entrenar modelos.
5. `validation` se usa para controlar entrenamiento, seleccionar hiperparámetros y ajustar el umbral de atipicidad.
6. `test` se reserva exclusivamente para la evaluación final.
7. El preprocesamiento se ajusta solo con `train` y luego se aplica a `validation` y `test`.
8. La evaluación principal se reporta a nivel vehículo/trayectoria.
9. La evaluación por ventana se conserva como análisis secundario para interpretar la evolución temporal de los scores.
10. Los notebooks no son pasos manuales obligatorios; el flujo oficial se ejecuta desde `main.py`.

---

## 3. Arquitectura general

```text
Google Drive                                      Google Colab GPU
TFM_SCANIA/                                       main.py
├── data/raw/       ───────────────────────►     ScaniaOutlierPipeline
├── data/processed/ ◄───────────────────────     EDA + Preprocessing + Windowing
├── models/         ◄───────────────────────     Training
├── outputs/        ◄───────────────────────     Evaluation + Comparison
└── doc/            ◄───────────────────────     Reportes, anexos y defensa
```

Flujo lógico:

```text
data/raw
   ↓
EDA y calidad de datos
   ↓
preprocesamiento ajustado solo con train
   ↓
construcción de ventanas temporales
   ↓
entrenamiento de modelos en Colab GPU
   ↓
cálculo de outlier scores
   ↓
ajuste de umbral con validation
   ↓
evaluación final con test
   ↓
outputs, métricas, predicciones y modelos guardados en Drive
```

---

## 4. Estructura del repositorio

```text
scania-component-x-outlier-detection-final-colab/
│
├── main.py
├── requirements.txt
├── pyproject.toml
├── README.md
│
├── config/
│   ├── config.colab.yaml
│   ├── config.debug.yaml
│   └── config.full.yaml
│
├── notebooks/
│   ├── 00_run_full_pipeline_colab.ipynb
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_visual_results.ipynb
│   └── 03_model_interpretation.ipynb
│
├── scripts/
│   ├── check_environment.py
│   ├── create_drive_folders.py
│   └── compare_metrics.py
│
├── src/
│   └── scania_outliers/
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py
│       ├── data_quality.py
│       ├── labels.py
│       ├── preprocessing.py
│       ├── windowing.py
│       ├── outlier_detection.py
│       ├── vehicle_level.py
│       ├── model_factory.py
│       ├── model_evaluation.py
│       ├── temporal_analysis.py
│       ├── pipelines/
│       │   ├── context.py
│       │   ├── orchestrator.py
│       │   └── stages.py
│       ├── models/
│       │   └── autoencoders.py
│       ├── training/
│       │   └── trainer.py
│       └── utils/
│           └── reproducibility.py
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── execution_guide_colab.md
│   └── diagrams/
│
└── tests/
    ├── test_cli.py
    ├── test_labels.py
    ├── test_windowing.py
    ├── test_model_evaluation.py
    ├── test_outlier_detection.py
    ├── test_model_factory.py
    └── test_vehicle_level.py
```

---

## 5. Estructura esperada en Google Drive

Crear la siguiente estructura en Drive:

```text
MyDrive/
└── TFM_SCANIA/
    ├── data/
    │   ├── raw/
    │   │   ├── train_operational_readouts.csv
    │   │   ├── train_tte.csv
    │   │   ├── train_specifications.csv
    │   │   ├── validation_operational_readouts.csv
    │   │   ├── validation_labels.csv
    │   │   ├── validation_specifications.csv
    │   │   ├── test_operational_readouts.csv
    │   │   ├── test_labels.csv
    │   │   └── test_specifications.csv
    │   ├── processed/
    │   └── samples/
    ├── models/
    ├── outputs/
    ├── experiments/
    └── doc/
```

Ruta principal esperada por el proyecto:

```text
/content/drive/MyDrive/TFM_SCANIA/data/raw
```

También se mantiene compatibilidad con una ruta alternativa heredada:

```text
/content/drive/MyDrive/TFM_SCANIA/raw
```

---

## 6. Preparación en Google Colab

### 6.1 Activar GPU

En Colab:

```text
Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU
```

Verificar GPU:

```bash
!nvidia-smi
```

### 6.2 Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 6.3 Ir al proyecto

Si el proyecto está guardado en Drive:

```bash
%cd /content/drive/MyDrive/TFM_SCANIA/project/scania-component-x-outlier-detection-final-colab
```

Si se clona desde GitHub:

```bash
!git clone https://github.com/USUARIO/scania-component-x-outlier-detection-final-colab.git /content/scania-component-x-outlier-detection-final-colab
%cd /content/scania-component-x-outlier-detection-final-colab
```

### 6.4 Instalar dependencias

```bash
!pip install -r requirements.txt
!pip install -e .
```

Si PySpark requiere Java:

```bash
!apt-get update -qq
!apt-get install -y openjdk-11-jdk-headless -qq
!pip install pyspark
```

---

## 7. Validación inicial

Validar entorno y rutas:

```bash
python scripts/check_environment.py --config config/config.colab.yaml
```

Crear carpetas necesarias en Drive:

```bash
python scripts/create_drive_folders.py --config config/config.colab.yaml
```

Validar configuración sin ejecutar procesos pesados:

```bash
python main.py \
  --config config/config.colab.yaml \
  --stage all \
  --model all \
  --dry-run
```

---

## 8. Ejecución rápida

### 8.1 Modo debug

Ejecuta el pipeline con un subconjunto de vehículos. Sirve para verificar que el flujo funcione antes de lanzar el entrenamiento completo.

```bash
python main.py \
  --config config/config.colab.yaml \
  --stage all \
  --model all \
  --mode debug
```

### 8.2 Modo full

Ejecuta el pipeline completo para generar resultados finales del TFM.

```bash
python main.py \
  --config config/config.colab.yaml \
  --stage all \
  --model all \
  --mode full
```

---

## 9. Ejecución por etapas recomendada

Para evitar pérdida de trabajo por desconexiones de Colab, se recomienda ejecutar por etapas. Cada etapa guarda artefactos en Google Drive.

### 9.1 EDA

```bash
python main.py --config config/config.colab.yaml --stage eda --mode full
```

### 9.2 Preprocesamiento y construcción de ventanas

```bash
python main.py --config config/config.colab.yaml --stage preprocess --mode full
```

Esta etapa genera:

```text
TFM_SCANIA/data/processed/train_windows.npz
TFM_SCANIA/data/processed/validation_windows.npz
TFM_SCANIA/data/processed/test_windows.npz
TFM_SCANIA/outputs/runs/<run_id>/manifests/preprocessing_metadata.json
```

### 9.3 Entrenar todos los modelos

```bash
python main.py --config config/config.colab.yaml --stage train --model all --mode full
```

### 9.4 Entrenar un modelo específico

```bash
python main.py --config config/config.colab.yaml --stage train --model lstm_autoencoder --mode full
python main.py --config config/config.colab.yaml --stage train --model cnn_lstm_autoencoder --mode full
python main.py --config config/config.colab.yaml --stage train --model transformer_encoder_simplified --mode full
```

### 9.5 Evaluar todos los modelos

```bash
python main.py --config config/config.colab.yaml --stage evaluate --model all --mode full
```

### 9.6 Consolidar comparación final

```bash
python main.py --config config/config.colab.yaml --stage compare --mode full
```

---

## 10. Comandos finales recomendados para el TFM

```bash
python main.py --config config/config.colab.yaml --stage eda --mode full
python main.py --config config/config.colab.yaml --stage preprocess --mode full
python main.py --config config/config.colab.yaml --stage train --model lstm_autoencoder --mode full
python main.py --config config/config.colab.yaml --stage train --model cnn_lstm_autoencoder --mode full
python main.py --config config/config.colab.yaml --stage train --model transformer_encoder_simplified --mode full
python main.py --config config/config.colab.yaml --stage evaluate --model all --mode full
python main.py --config config/config.colab.yaml --stage compare --mode full
```

---

## 11. Salidas generadas

```text
TFM_SCANIA/
├── data/processed/
│   ├── train_windows.npz
│   ├── validation_windows.npz
│   └── test_windows.npz
├── models/
│   ├── lstm_autoencoder/model.pt
│   ├── cnn_lstm_autoencoder/model.pt
│   └── transformer_encoder_simplified/model.pt
└── outputs/
    ├── metrics/
    ├── predictions/
    ├── figures/
    ├── tables/
    └── runs/<run_id>/manifests/
```

---

## 12. Evaluación

La evaluación se realiza en dos niveles.

### 12.1 Nivel ventana

```text
vehicle_id
start_time
end_time
outlier_score
is_outlier
y_true
```

Este nivel sirve para analizar la evolución temporal de los scores.

### 12.2 Nivel vehículo/trayectoria

```text
vehicle_id
y_true
max_score
mean_score
p95_score
outlier_window_ratio
is_outlier
```

Este es el nivel principal para reportar resultados cuando las etiquetas disponibles se encuentran asociadas al vehículo o trayectoria.

Métricas consideradas:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

---

## 13. Pruebas

Ejecutar pruebas unitarias:

```bash
pytest
```

---

## 14. Limitaciones metodológicas

- La evaluación depende de la granularidad de las etiquetas disponibles.
- Si las etiquetas están a nivel vehículo, las métricas por ventana deben interpretarse solo como apoyo exploratorio.
- Si por restricciones de Colab se trabaja con una muestra, debe documentarse el número de vehículos y el criterio de selección.
- El Transformer Encoder se implementa en versión simplificada para mantener el alcance viable del TFM.
- No se realizan nuevas particiones ni validación cruzada porque se respetan los conjuntos oficiales del dataset.

---

## 15. Resumen operativo

```text
1. Subir CSV a Drive/data/raw
2. Montar Drive en Colab
3. Instalar dependencias
4. Validar entorno
5. Ejecutar debug
6. Ejecutar full por etapas
7. Revisar outputs/metrics y outputs/predictions
8. Usar notebooks solo para visualización y defensa
```
