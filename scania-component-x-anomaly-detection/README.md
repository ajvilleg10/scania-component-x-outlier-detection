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
11. El dataset descargado con KaggleHub debe copiarse a `Google Drive/data/raw` antes de ejecutar el pipeline.
12. El pipeline principal no utiliza Pandas ni `toPandas()`; Spark procesa datos y genera ventanas en Parquet.
13. PyTorch consume las ventanas desde Parquet por batches para entrenar en GPU sin cargar todo el dataset en memoria.

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
validación de archivos requeridos
   ↓
EDA y calidad de datos
   ↓
preprocesamiento ajustado solo con train usando Spark
   ↓
construcción de ventanas temporales en Spark
   ↓
guardado de ventanas en Parquet particionado
   ↓
PyTorch lee Parquet por batches
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
│   ├── download_kaggle_to_drive.py
│   ├── check_raw_files.py
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
│       ├── spark_windowing.py
│       ├── windowing.py              # solo pruebas/ejemplos pequeños
│       ├── outlier_detection.py
│       ├── vehicle_level.py
│       ├── model_factory.py
│       ├── model_evaluation.py
│       ├── temporal_analysis.py
│       ├── datasets.py               # lectura streaming de Parquet para PyTorch
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

El pipeline no lee directamente desde la caché temporal de KaggleHub.

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
```

Si PySpark requiere Java:

```bash
!apt-get update -qq
!apt-get install -y openjdk-11-jdk-headless -qq
!pip install pyspark
```

---

## 7. Descarga del dataset desde Kaggle

El dataset se descarga con `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("tapanbatla/scania-component-x-dataset-2025")
```

Sin embargo, KaggleHub lo almacena primero en la caché temporal de Colab, normalmente bajo:

```text
/root/.cache/kagglehub/datasets/...
```

Esa ruta no es persistente y no corresponde a la estructura del proyecto. Por eso, antes de ejecutar `main.py`, los CSV deben copiarse obligatoriamente a:

```text
/content/drive/MyDrive/TFM_SCANIA/data/raw
```

Para hacerlo automáticamente, ejecute:

```bash
python scripts/download_kaggle_to_drive.py \
  --config config/config.colab.yaml
```

O indicando explícitamente dataset y destino:

```bash
python scripts/download_kaggle_to_drive.py \
  --dataset tapanbatla/scania-component-x-dataset-2025 \
  --raw-dir /content/drive/MyDrive/TFM_SCANIA/data/raw
```

Después valide que todos los archivos requeridos estén disponibles:

```bash
python scripts/check_raw_files.py \
  --config config/config.colab.yaml
```

---

## 8. Ejecución rápida

### 8.1 Crear carpetas en Drive

```bash
python scripts/create_drive_folders.py \
  --config config/config.colab.yaml
```

### 8.2 Descargar/copiar dataset a Drive/raw

```bash
python scripts/download_kaggle_to_drive.py \
  --config config/config.colab.yaml
```

### 8.3 Validar archivos raw

```bash
python scripts/check_raw_files.py \
  --config config/config.colab.yaml
```

### 8.4 Validar pipeline sin procesos pesados

```bash
python main.py \
  --config config/config.colab.yaml \
  --stage all \
  --model all \
  --mode debug \
  --dry-run
```

### 8.5 Ejecutar prueba en modo debug

```bash
python main.py \
  --config config/config.colab.yaml \
  --stage all \
  --model all \
  --mode debug
```

### 8.6 Ejecutar experimento final

```bash
python main.py \
  --config config/config.colab.yaml \
  --stage all \
  --model all \
  --mode full
```

---

## 9. Ejecución por etapas

Si Colab se desconecta o se desea controlar mejor el proceso, se recomienda ejecutar por etapas.

### Validar datos

```bash
python main.py --config config/config.colab.yaml --stage check-data --mode full
```

### EDA

```bash
python main.py --config config/config.colab.yaml --stage eda --mode full
```

### Preprocesamiento y windowing

```bash
python main.py --config config/config.colab.yaml --stage preprocess --mode full
```

### Entrenar modelos

```bash
python main.py --config config/config.colab.yaml --stage train --model all --mode full
```

Entrenar un solo modelo:

```bash
python main.py --config config/config.colab.yaml --stage train --model lstm_autoencoder --mode full
```

```bash
python main.py --config config/config.colab.yaml --stage train --model cnn_lstm_autoencoder --mode full
```

```bash
python main.py --config config/config.colab.yaml --stage train --model transformer_encoder_simplified --mode full
```

### Evaluar

```bash
python main.py --config config/config.colab.yaml --stage evaluate --model all --mode full
```

### Comparar

```bash
python main.py --config config/config.colab.yaml --stage compare --mode full
```

---

## 10. Comando recomendado real en Colab

En la práctica, se recomienda ejecutar:

```bash
python scripts/create_drive_folders.py --config config/config.colab.yaml
python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml
python scripts/check_raw_files.py --config config/config.colab.yaml
python main.py --config config/config.colab.yaml --stage all --model all --mode debug
```

Si todo funciona correctamente:

```bash
python main.py --config config/config.colab.yaml --stage all --model all --mode full
```

---

## 11. Salidas esperadas

```text
/content/drive/MyDrive/TFM_SCANIA/data/processed/windows/train/
/content/drive/MyDrive/TFM_SCANIA/data/processed/windows/validation/
/content/drive/MyDrive/TFM_SCANIA/data/processed/windows/test/
/content/drive/MyDrive/TFM_SCANIA/models/
/content/drive/MyDrive/TFM_SCANIA/outputs/
/content/drive/MyDrive/TFM_SCANIA/experiments/
```

Resultados principales:

```text
outputs/metrics/
outputs/predictions/
outputs/comparisons/
outputs/figures/
outputs/runs/<run_id>/manifests/
```

---

## 11.1. Procesamiento sin Pandas

La versión corregida del proyecto evita `toPandas()` en el pipeline principal. Esto mitiga errores de Arrow en Google Colab y evita trasladar DataFrames grandes al driver de Python.

Flujo técnico aplicado:

```text
CSV en Drive/data/raw
   ↓
Spark DataFrames
   ↓
EDA, calidad, imputación y escalado en Spark
   ↓
SparkWindowBuilder genera ventanas temporales
   ↓
Parquet particionado en data/processed/windows/
   ↓
PyTorch lee Parquet por batches con PyArrow
   ↓
Entrenamiento y evaluación
```

Por esta razón, `config.colab.yaml` usa:

```yaml
spark:
  arrow_enabled: false

windowing:
  output_format: parquet
  use_pandas: false
```

---

## 12. Notebooks

Los notebooks no son el flujo oficial de ejecución. Se usan para:

- montar Drive desde Colab;
- lanzar comandos del pipeline;
- visualizar resultados;
- generar figuras para el TFM;
- apoyar la defensa.

El notebook principal es:

```text
notebooks/00_run_full_pipeline_colab.ipynb
```

---

## 13. Consideraciones importantes

- No modificar manualmente las particiones oficiales del dataset.
- No mezclar `train`, `validation` y `test`.
- No ajustar el escalador ni la imputación con `validation` o `test`.
- No seleccionar umbral con `test`.
- No evaluar resultados finales con `validation`.
- No leer los CSV desde `/root/.cache/kagglehub/`.
- No usar `toPandas()` sobre los DataFrames de lecturas operacionales.
- Mantener `spark.arrow_enabled: false` en Colab para evitar errores Arrow/Java.
- Generar ventanas en Parquet, no en `.npz`, para evitar cargar todo en memoria.
- Copiar siempre los CSV a `Drive/data/raw` antes de ejecutar `main.py`.
- Reportar resultados finales únicamente con `test`.

---

## 14. Pruebas

```bash
pytest -q
```

---

## 15. Licencia

Uso académico para el desarrollo del TFM.
