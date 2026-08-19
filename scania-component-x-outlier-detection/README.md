# SCANIA Component X — Detección de outliers temporales multivariados

Proyecto experimental del TFM orientado a comparar **LSTM Autoencoder**, **CNN-LSTM Autoencoder** y **Transformer Encoder simplificado** sobre las series temporales multivariadas del dataset SCANIA Component X.

La arquitectura separa de forma explícita el **código versionado en Git** de los **datos y artefactos experimentales persistidos en Google Drive**. El procesamiento se realiza con PySpark, las ventanas se almacenan en Parquet y el modelado se realiza con PyTorch. El pipeline principal no utiliza `toPandas()`.

## 1. Principios del proyecto

- Se respetan las particiones oficiales `train`, `validation` y `test`; no se crea un nuevo `train_test_split` ni se aplica validación cruzada.
- El preprocesamiento se ajusta únicamente con `train`.
- `validation` se emplea para control del entrenamiento y selección del umbral de outlier.
- `test` queda reservado para la evaluación final.
- Para la evaluación binaria del TFM, `class_label=0` se usa como grupo de referencia negativo y las clases temporales oficiales `1..4` se agrupan como grupo positivo; las cinco clases originales se conservan en el EDA.
- Los modelos se ejecutan **uno por uno** en Colab.
- `--prepare-only` se ejecuta una sola vez por experimento y **no requiere `--model`**.
- `--run-name` identifica y separa cada experimento (`debug_025`, `debug_050`, `debug_100`, `debug_200`, `full`).
- Cada run conserva sus modelos, métricas, predicciones, figuras, tablas, logs, configuración y manifiestos.
- Las ventanas Parquet son un **working cache compartido** en `data/processed/windows`; antes de entrenar se valida que correspondan al `run-name` activo.

## 2. Estructura del repositorio Git

```text
scania-component-x-outlier-detection/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── CHANGELOG_TFM.md
├── config/
├── scripts/
├── src/
├── tests/
├── notebooks/
└── docs/
```

El repositorio **no contiene** `data/`, `models/`, `outputs/` ni `reports/`. Esos artefactos se generan en Drive y no deben versionarse en Git.

## 3. Estructura en Google Drive

El proyecto crea solo las carpetas necesarias. Las subcarpetas de resultados se crean cuando realmente contienen artefactos.

```text
/content/drive/MyDrive/TFM_SCANIA/
├── data/
│   ├── raw/
│   └── processed/
│       ├── windows/
│       │   ├── train/
│       │   ├── validation/
│       │   └── test/
│       ├── metadata/
│       └── manifests/
└── experiments/
    ├── runs/
    │   ├── debug_025/
    │   ├── debug_050/
    │   ├── debug_100/
    │   ├── debug_200/
    │   └── full/
    └── study_summary/   # se crea solo al ejecutar --study-summary
```

Una ejecución completa queda, por ejemplo, así:

```text
experiments/runs/debug_025/
├── config_used.yaml
├── runner_events.jsonl
├── models/
│   ├── lstm_autoencoder/
│   ├── cnn_lstm_autoencoder/
│   └── transformer_encoder/
├── metrics/
├── predictions/
├── comparisons/
├── figures/
│   ├── eda/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── comparison/
├── tables/
│   ├── eda/
│   ├── preprocessing/
│   └── report/
├── logs/
└── manifests/
```

## 4. Figuras generadas

El pipeline genera automáticamente evidencia visual útil para el TFM.

**EDA:** distribución original de las cinco clases temporales, distribución binaria usada en la evaluación, distribuciones de variables, box plots, matriz de correlación de variables seleccionadas, distribución de lecturas por vehículo y gráficos de valores faltantes cuando se calcula el reporte completo.

**Preprocesamiento:** resumen de variables seleccionadas/descartadas y número de ventanas generadas por partición.

**Entrenamiento:** curvas `train_loss` y `validation_loss` por modelo.

**Evaluación:** matriz de confusión, curva Precision-Recall, curva ROC, distribución de outlier scores y box plot de scores por clase, calculados con la referencia final a nivel vehículo. A nivel ventana se guardan scores y predicciones para análisis temporal, pero no Precision/Recall/F1 supervisados porque no existen etiquetas de anomalía por ventana.

**Comparación:** comparación conjunta de Precision, Recall, F1, PR-AUC y ROC-AUC y comparación de tiempos de entrenamiento/inferencia.

Las figuras de EDA sobre las variables operacionales usan una muestra acotada y configurable para visualización. Esa muestra **no se utiliza para calcular las métricas finales de los modelos**.

La conversión de `class_label` a referencia binaria se documenta explícitamente: la clase 0 representa lecturas situadas a más de 48 `time_step` del fallo, mientras que las clases 1–4 corresponden a ventanas progresivamente más próximas al fallo y se agrupan como positivas para la evaluación de detección de outliers. Esta referencia sirve para contrastar los scores del modelo y no implica que el dataset proporcione etiquetas puntuales de anomalía para cada lectura.

## 5. Preparación de Google Colab

Active una GPU desde la interfaz de Colab y ejecute:

```python
!nvidia-smi
```

Instale Java 17:

```python
!apt-get update -qq
!apt-get install -y openjdk-17-jdk-headless -qq

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

!java -version
```

Monte Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone el repositorio:

```python
%cd /content
!rm -rf scania-component-x-outlier-detection
!git clone https://github.com/TU_USUARIO/scania-component-x-outlier-detection.git
%cd /content/scania-component-x-outlier-detection
```

Instale dependencias:

```python
!pip install -q -r requirements.txt
```

## 6. Flujo profesional de un experimento

### 6.1 Preparar `debug_025` una sola vez

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --prepare-only
```

`--prepare-only` crea/verifica las carpetas compartidas, reutiliza los CSV si ya existen, descarga desde KaggleHub solo cuando faltan, valida los nueve archivos, ejecuta EDA, preprocesa y genera las ventanas Parquet. No entrena ningún modelo.

### 6.2 LSTM Autoencoder

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --model lstm_autoencoder
```

### 6.3 CNN-LSTM Autoencoder

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --model cnn_lstm_autoencoder
```

### 6.4 Transformer Encoder

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --model transformer_encoder
```

Cada comando de modelo ejecuta **entrenamiento + evaluación** usando las ventanas ya preparadas. No se necesitan `--skip-download`, `--skip-eda` ni `--skip-preprocess`.

### 6.5 Comparar los tres modelos

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --compare-only
```

La comparación exige, por defecto, métricas a nivel vehículo de los tres modelos.

### 6.6 Cerrar el run con el resumen de artefactos

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --report-only
```

## 7. Repetir el background experimental

Repita exactamente el mismo flujo cambiando el `--run-name` (`debug_025`, `debug_050`, `debug_100` o `debug_200`).

```text
debug_025  -> 25 vehículos
debug_050  -> 50 vehículos
debug_100  -> 100 vehículos
debug_200  -> 200 vehículos
full       -> todos los datos
```

Cuando el `run-name` sigue el patrón `debug_NNN`, el runner infiere automáticamente el número de vehículos. Por ello, `--max-vehicles` es opcional en el flujo recomendado; si se proporciona, el script valida que sea coherente con el nombre del run.

Ejemplo para preparar 50 vehículos:

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_050 \
  --prepare-only
```

Después ejecute los tres modelos, `--compare-only` y `--report-only` con el mismo `debug_050`.

## 8. Ejecución final `full`

Prepare:

```bash
python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --prepare-only
```

Entrene/evalúe cada modelo de forma independiente:

```bash
python scripts/run_colab_safe.py --config config/config.full.yaml --mode full --run-name full --model lstm_autoencoder
python scripts/run_colab_safe.py --config config/config.full.yaml --mode full --run-name full --model cnn_lstm_autoencoder
python scripts/run_colab_safe.py --config config/config.full.yaml --mode full --run-name full --model transformer_encoder
```

Compare y cierre el run:

```bash
python scripts/run_colab_safe.py --config config/config.full.yaml --mode full --run-name full --compare-only
python scripts/run_colab_safe.py --config config/config.full.yaml --mode full --run-name full --report-only
```

### 8.1 Consolidar el background completo del TFM

Después de disponer de los runs `debug_025`, `debug_050`, `debug_100`, `debug_200` y `full`, genere una síntesis transversal:

```bash
python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --study-summary
```

El comando crea `experiments/study_summary/` con una tabla de métricas de todos los runs, el mejor modelo por ejecución y gráficas de evolución de PR-AUC, F1-score y Recall. Esta síntesis documenta el desarrollo progresivo; la interpretación principal del TFM sigue correspondiendo a `full`.

## 9. Recuperación de etapas

El flujo normal no necesita flags de salto. Para recuperación se mantienen dos comandos especializados:

```bash
# Solo reentrenar
python scripts/run_colab_safe.py ... --model lstm_autoencoder --train-only

# Solo reevaluar un modelo ya entrenado
python scripts/run_colab_safe.py ... --model lstm_autoencoder --evaluate-only
```

## 10. Protección frente a mezclas de runs

`data/processed/windows` funciona como caché de trabajo. Al terminar `--prepare-only`, el proyecto registra el `run-name`, el modo, el tamaño debug y una huella de la configuración. Antes de entrenar o evaluar, se valida esa huella. Si las ventanas activas pertenecen a otro experimento, el proceso se detiene y solicita volver a ejecutar `--prepare-only`.

Esto evita, por ejemplo, entrenar accidentalmente `debug_025` sobre ventanas generadas para `debug_100`.

## 11. Resultados del TFM

Los runs `debug_025`, `debug_050`, `debug_100` y `debug_200` sirven como evidencia del desarrollo progresivo, estabilidad y escalado del pipeline. En validation/test, el modo debug aplica una selección determinista aproximadamente estratificada y garantiza un mínimo pequeño de casos positivos para que las métricas puedan ejercitarse incluso con 25 vehículos. Por ello, estas métricas se interpretan como resultados de validación técnica, no como estimaciones finales. Las métricas de `full` constituyen la comparación experimental principal. Al mantenerse separados por `--run-name`, los resultados anteriores no se pierden cuando se ejecuta el siguiente tamaño.

## 12. Tests

```bash
pytest -q
```

Los tests cubren CLI, validación de datos, etiquetas, métricas, factory de modelos, thresholding, agregación por vehículo, estado de preparación, visualizaciones y consolidación entre runs.
