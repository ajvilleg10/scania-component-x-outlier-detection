# SCANIA Component X - Outlier Detection

Proyecto de TFM para la **detección de outliers temporales multivariados** en lecturas operacionales del dataset **SCANIA Component X**, orientado a un contexto de mantenimiento predictivo en vehículos pesados.

El proyecto está diseñado para ejecutarse en **Google Colab + Google Drive**. Spark se utiliza para la ingesta, validación, análisis exploratorio, preprocesamiento y construcción de ventanas temporales. PyTorch se utiliza para el entrenamiento de los modelos de aprendizaje profundo.

---

## 1. Objetivo del proyecto

Diseñar, implementar y evaluar tres modelos de aprendizaje profundo para detectar trayectorias operacionales atípicas en series temporales multivariadas:

- `lstm_autoencoder`
- `cnn_lstm_autoencoder`
- `transformer_encoder`

El análisis respeta las particiones oficiales del dataset:

| Partición | Uso |
|---|---|
| `train` | Ajuste del preprocesamiento y entrenamiento de modelos |
| `validation` | Control de entrenamiento, selección de umbral y ajuste de configuración |
| `test` | Evaluación final |

No se realiza `train_test_split`, `k-fold` ni validación cruzada adicional.

---

## 2. Principios de diseño

El proyecto sigue estas decisiones técnicas:

- El código fuente se ejecuta desde `/content` para evitar lentitud al ejecutar directamente desde Drive.
- Los datos, modelos, métricas y logs se guardan en Google Drive.
- El pipeline principal no usa `toPandas()` sobre datos grandes.
- Las ventanas temporales se generan con Spark y se guardan como Parquet particionado.
- Los modelos se entrenan **uno a uno** para reducir el riesgo de caídas en Colab.
- El conjunto `test` queda reservado exclusivamente para evaluación final.
- Todas las etapas guardan artefactos y logs para facilitar trazabilidad.

---

## 3. Estructura del proyecto

```text
scania-component-x-outlier-detection/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── CHANGELOG_TFM.md
├── config/
│   ├── config.colab.yaml
│   ├── config.debug.yaml
│   └── config.full.yaml
├── notebooks/
│   ├── 00_run_full_pipeline_colab.ipynb
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_visual_results.ipynb
│   └── 03_model_interpretation.ipynb
├── scripts/
│   ├── create_drive_folders.py
│   ├── download_kaggle_to_drive.py
│   ├── check_raw_files.py
│   ├── check_environment.py
│   ├── run_colab_safe.py
│   └── compare_metrics.py
├── src/scania_outliers/
│   ├── cli.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── preprocessing.py
│   ├── spark_session.py
│   ├── spark_windowing.py
│   ├── datasets.py
│   ├── labels.py
│   ├── model_factory.py
│   ├── model_evaluation.py
│   ├── outlier_detection.py
│   ├── vehicle_level.py
│   ├── pipelines/
│   ├── models/
│   └── training/
├── tests/
└── docs/
```

---

## 4. Estructura esperada en Google Drive

El proyecto espera esta estructura persistente:

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
├── models/
│   ├── lstm_autoencoder/
│   ├── cnn_lstm_autoencoder/
│   └── transformer_encoder/
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   ├── comparisons/
│   ├── logs/
│   └── runs/
├── experiments/
│   └── runs/
│       ├── debug_025/
│       ├── debug_050/
│       ├── debug_100/
│       ├── debug_200/
│       └── full/
└── doc/
```

---

## 5. Archivos requeridos en `data/raw`

Antes de ejecutar el pipeline, estos archivos deben estar en:

```text
/content/drive/MyDrive/TFM_SCANIA/data/raw
```

```text
train_operational_readouts.csv
train_tte.csv
train_specifications.csv
validation_operational_readouts.csv
validation_labels.csv
validation_specifications.csv
test_operational_readouts.csv
test_labels.csv
test_specifications.csv
```

El script `download_kaggle_to_drive.py` descarga el dataset con KaggleHub y copia los CSV desde la caché temporal de Colab hacia `Drive/data/raw`.

---

## 6. Preparación en Google Colab

### 6.1. Activar GPU

En Colab:

```text
Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU
```

Verificar:

```bash
!nvidia-smi
```

### 6.2. Instalar Java 17

```python
!apt-get update -qq
!apt-get install -y openjdk-17-jdk-headless -qq

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

!java -version
```

### 6.3. Montar Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 6.4. Descomprimir el proyecto en `/content`

```bash
!rm -rf /content/scania-component-x-outlier-detection

!unzip -q "/content/drive/MyDrive/TFM_SCANIA/project/scania-component-x-outlier-detection-tfm-professional.zip" \
  -d /content/scania-component-x-outlier-detection
```

Entrar al proyecto:

```python
%cd /content/scania-component-x-outlier-detection/scania-component-x-outlier-detection
```

Verificar:

```bash
!ls
```

Deben aparecer `main.py`, `config`, `scripts`, `src` y `README.md`.

### 6.5. Instalar dependencias

```bash
!pip install -q -r requirements.txt
```

---

## 7. Ejecución recomendada

La ejecución recomendada en Colab es **por fases**, no todo de una sola vez.

### 7.1. Preparar datos una sola vez

Primero usar una muestra pequeña:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model lstm_autoencoder \
  --prepare-only
```

Esta fase ejecuta:

1. Creación de carpetas en Drive.
2. Descarga del dataset desde Kaggle si aplica.
3. Validación de archivos raw.
4. EDA en modo seguro.
5. Preprocesamiento con Spark.
6. Construcción de ventanas Parquet.

### 7.2. Entrenar y evaluar LSTM Autoencoder

```bash
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model lstm_autoencoder \
  --skip-download \
  --skip-eda \
  --skip-preprocess
```

### 7.3. Entrenar y evaluar CNN-LSTM Autoencoder

```bash
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model cnn_lstm_autoencoder \
  --skip-download \
  --skip-eda \
  --skip-preprocess
```

### 7.4. Entrenar y evaluar Transformer Encoder

```bash
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model transformer_encoder \
  --skip-download \
  --skip-eda \
  --skip-preprocess
```

### 7.5. Comparar resultados

```bash
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --compare-only
```

---

## 8. Escalado progresivo

No pasar directamente a `full`. Usar esta progresión:

```text
debug_025 → 25 vehículos
debug_050 → 50 vehículos
debug_100 → 100 vehículos
debug_200 → 200 vehículos
full      → todos los vehículos disponibles
```

Ejemplo:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 100 \
  --run-name debug_100 \
  --model lstm_autoencoder \
  --prepare-only \
  --skip-download
```

---

## 8.1. Conservación automática de resultados con `--run-name`

El argumento `--run-name` evita que los resultados de una corrida se pierdan cuando se ejecuta otra. Al finalizar cada fase, el runner copia los artefactos disponibles hacia:

```text
/content/drive/MyDrive/TFM_SCANIA/experiments/runs/<run-name>/
```

Ejemplo para `debug_025`:

```text
experiments/runs/debug_025/
├── config_used.yaml
├── run_metadata.json
├── run_events.jsonl
├── logs/
├── metrics/
├── comparisons/
├── predictions/
├── figures/
├── tables/
├── models/
└── processed/
    ├── metadata/
    └── manifests/
```

Por defecto no se copian las ventanas Parquet para no duplicar demasiado espacio en Drive. Si se desea archivar también las ventanas, se puede añadir `--archive-windows`, aunque no se recomienda para cada debug por tamaño.


## 9. Ejecución en modo full

En modo full se recomienda ejecutar **un modelo por corrida**.

Preparar datos completos:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model lstm_autoencoder \
  --prepare-only \
  --skip-download
```

Entrenar LSTM:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model lstm_autoencoder \
  --skip-download \
  --skip-eda \
  --skip-preprocess
```

Entrenar CNN-LSTM:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model cnn_lstm_autoencoder \
  --skip-download \
  --skip-eda \
  --skip-preprocess
```

Entrenar Transformer:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model transformer_encoder \
  --skip-download \
  --skip-eda \
  --skip-preprocess
```

Comparar:

```bash
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --compare-only
```

---

## 10. Comandos directos por etapa

También se puede usar `main.py` directamente:

```bash
python main.py --config config/config.colab.yaml --stage check-data --mode debug
python main.py --config config/config.colab.yaml --stage eda --mode debug
python main.py --config config/config.colab.yaml --stage preprocess --mode debug
python main.py --config config/config.colab.yaml --stage train --model lstm_autoencoder --mode debug
python main.py --config config/config.colab.yaml --stage evaluate --model lstm_autoencoder --mode debug
python main.py --config config/config.colab.yaml --stage compare --model all --mode debug
```

Por seguridad, el CLI bloquea `--model all` en etapas pesadas si no se pasa explícitamente `--allow-all-models`.

---

## 11. Artefactos generados

| Carpeta | Contenido |
|---|---|
| `data/processed/windows/` | Ventanas Parquet por partición |
| `models/<modelo>/` | Pesos del modelo y umbral seleccionado |
| `outputs/metrics/` | Métricas JSON/CSV |
| `outputs/predictions/` | Predicciones por ventana y vehículo |
| `outputs/logs/` | Logs de ejecución segura más reciente |
| `outputs/comparisons/` | Comparaciones generadas por el stage `compare` |
| `outputs/runs/` | Manifiestos de etapas individuales |
| `experiments/runs/<run-name>/` | Copia archivada de resultados por corrida |

---

## 12. Buenas prácticas aplicadas

- Separación clara entre código, datos y resultados.
- Configuración externa en YAML.
- Ejecución modular por etapas.
- Entrenamiento de un modelo por corrida.
- Evita `toPandas()` en datos grandes.
- Preprocesamiento ajustado únicamente con `train`.
- Umbral seleccionado con `validation`.
- Evaluación final con `test`.
- Logs, manifiestos y archivo automático por `--run-name` para trazabilidad.
- Ventanas guardadas en formato Parquet particionado.
- Pruebas unitarias básicas en `tests/`.

---

## 13. Troubleshooting rápido

### Error: `ConnectionRefusedError` o `Py4JNetworkError`

Spark probablemente se quedó sin memoria o la JVM cayó. Reiniciar runtime y bajar `--max-vehicles`.

### Ejecución muy lenta

Usar `--prepare-only` una sola vez y luego `--skip-preprocess` para entrenar modelos sin regenerar ventanas.

### No encuentra archivos CSV

Ejecutar:

```bash
python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml
python scripts/check_raw_files.py --config config/config.colab.yaml
```

### No hay GPU

Verificar:

```bash
nvidia-smi
```

Si no aparece GPU, activar GPU en Colab o entrenar con menos épocas.

---

## 14. Pruebas

En local o Colab:

```bash
pytest
```

---

## 15. Nota metodológica

Por restricciones de memoria y tiempo de ejecución en Google Colab, los modelos se entrenan de forma independiente. Esta decisión no afecta la comparabilidad, porque los tres modelos utilizan las mismas ventanas, las mismas particiones oficiales y los mismos criterios de evaluación.
