# Guía de ejecución en Colab — v3.2.0

Reemplaza la guía anterior. Cambios de fondo respecto a la que ya conoces:
scoring calibrado por variable, `p95_score`, winsorización (v3.1.0); estudio
de curva de aprendizaje para la hipótesis 1 (v3.2.0); y ya **no se ejecutan**
`debug_050`, `debug_100` ni `debug_200` (no aportaban evidencia — ver
conversación anterior).

Secuencia definitiva:

```text
0. Actualizar el repo de GitHub con este código (v3.2.0)
1. Entorno (GPU, Java 17, Drive, clone, install)
2. Limpieza de Drive (residuo de versiones anteriores + runs viejos)
3. debug_025 — prueba de humo del código nuevo
4. Estudio de curva de aprendizaje (hipótesis 1)
5. full — resultado experimental principal
6. Consolidación final (study-summary + learning-curve-summary)
```

---

## 0. Antes de todo: actualizar el repositorio

El `git clone` de abajo trae lo que esté en tu GitHub. Si el ZIP que te
entregué (v3.2.0) todavía no está subido, reemplaza el contenido de tu repo
local con el del ZIP y haz `git add -A && git commit -m "v3.2.0" && git push`
**antes** de continuar. Si no haces esto, el clone de Colab va a traer código
viejo sin que te des cuenta.

## 1. Entorno de ejecución

```python
!nvidia-smi
!free -h
```

```python
!apt-get update -qq
!apt-get install -y openjdk-17-jdk-headless -qq

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

!java -version
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd /content
!rm -rf /content/scania-component-x-outlier-detection
!git clone https://github.com/TU_USUARIO/scania-component-x-outlier-detection.git
%cd /content/scania-component-x-outlier-detection
!ls -lah
```

```python
!pip install -q -r requirements.txt
```

```python
import torch, pyspark, pyarrow
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("PySpark:", pyspark.__version__)
print("PyArrow:", pyarrow.__version__)
```

```python
!python scripts/check_environment.py --config config/config.colab.yaml
```

```python
!pytest -q
```

Deberían pasar 24 tests (antes eran menos — se agregaron los de scoring
calibrado y curva de aprendizaje).

## 2. Limpieza de Drive

Borra residuo de versiones anteriores del proyecto y las corridas que no se
van a rehacer. Se va a la papelera de Drive, no es borrado permanente.

```python
BASE = "/content/drive/MyDrive/TFM_SCANIA"

import shutil
from pathlib import Path

to_delete = [
    f"{BASE}/models",                          # top-level, versión anterior sin usar
    f"{BASE}/outputs",                         # top-level, versión anterior sin usar
    f"{BASE}/data/processed/quality_reports",
    f"{BASE}/data/processed/clean",
    f"{BASE}/data/processed/train_windows.npz",
    f"{BASE}/data/processed/test_windows.npz",
    f"{BASE}/data/processed/validation_windows.npz",
    f"{BASE}/experiments/registry",
    f"{BASE}/experiments/runs/debug_050",      # no se rehace
    f"{BASE}/experiments/runs/debug_100",      # no se rehace
    f"{BASE}/experiments/runs/debug_200",      # no se rehace
    f"{BASE}/experiments/runs/debug_025",      # se rehace con v3.2.0 en el paso 3
    f"{BASE}/experiments/runs/full",           # se rehace con v3.2.0 en el paso 5
    f"{BASE}/experiments/study_summary",       # se regenera al final
]

for path in to_delete:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        print("Borrado:", path)
    else:
        print("No existía:", path)

print("\ndata/raw y data/processed/{windows,metadata,manifests} NO se tocaron.")
```

## 3. `debug_025` — prueba de humo del código v3.2.0

```python
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --prepare-only
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --model lstm_autoencoder
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --model cnn_lstm_autoencoder
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --model transformer_encoder
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --compare-only

!python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --report-only
```

Verificación rápida — deben aparecer `feature_error_scale.json` por modelo y
columnas `clip_low`/`clip_high` en la tabla de preprocesamiento:

```python
!find /content/drive/MyDrive/TFM_SCANIA/experiments/runs/debug_025 -name "feature_error_scale.json"
!cat /content/drive/MyDrive/TFM_SCANIA/experiments/runs/debug_025/tables/preprocessing/train_fitted_parameters.csv | head -5
```

Si esto corrió limpio de punta a punta, el código está validado. Continúa.

## 4. Estudio de curva de aprendizaje (hipótesis específica 1)

```python
!python scripts/run_learning_curve.py \
  --config config/config.full.yaml \
  --sizes 25,50,100,200 \
  --n-seeds 3 \
  --model lstm_autoencoder
```

Son 12 combinaciones (4 tamaños × 3 semillas); cada una tarda segundos a
minutos de entrenamiento. Corre prepare→train→evaluate→report por cada una,
igual que un run normal, solo que automatizado en un solo comando.

No consolides todavía — el resumen final necesita que `full` (paso 5) también
esté listo, porque sirve de ancla en el extremo de la curva.

## 5. `full` — resultado experimental principal

```python
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --prepare-only
```

Esta fase puede tardar bastante más que las anteriores. Déjala terminar.

```python
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model lstm_autoencoder
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model cnn_lstm_autoencoder
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --model transformer_encoder
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --compare-only

!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --mode full \
  --run-name full \
  --report-only
```

## 6. Consolidación final

```python
!python scripts/build_learning_curve_summary.py --config config/config.full.yaml
```

```python
!python scripts/run_colab_safe.py \
  --config config/config.full.yaml \
  --study-summary
```

(esto último solo va a encontrar `debug_025` y `full` en `experiments/runs/`,
ya que `debug_050/100/200` se borraron y no se rehicieron — es justo lo
esperado).

```python
!find /content/drive/MyDrive/TFM_SCANIA/experiments/learning_curve_summary -type f | sort
!find /content/drive/MyDrive/TFM_SCANIA/experiments/study_summary -type f | sort
!find /content/drive/MyDrive/TFM_SCANIA/experiments/runs/full/figures -type f | sort
```

## Reglas que se mantienen igual que antes

- Un modelo a la vez, nunca `--model all` en train/evaluate.
- `--prepare-only` una vez por `--run-name` (o de nuevo si cambiaste de
  `run-name` y necesitas volver a uno anterior — regenera la caché
  compartida de ventanas).
- Si Colab se desconecta: el código en `/content` desaparece, lo de Drive
  permanece. Repite entorno (pasos 1) y continúa exactamente en el comando
  donde ibas — no repitas `--prepare-only` si la ventana activa sigue
  perteneciendo al mismo `run-name`.
- `--train-only` / `--evaluate-only` son solo para recuperación puntual, no
  para el flujo normal.

## Estructura final esperada en Drive

```text
TFM_SCANIA/
├── data/
│   ├── raw/                          9 CSV originales
│   └── processed/
│       ├── windows/                  caché compartida (última preparación)
│       ├── metadata/
│       └── manifests/
├── experiments/
│   ├── runs/
│   │   ├── debug_025/
│   │   ├── learning_curve_n025_s1/ ... learning_curve_n200_s3/   (12 carpetas)
│   │   └── full/
│   ├── study_summary/                debug_025 + full
│   └── learning_curve_summary/       tabla + 4 figuras
└── doc/                               tu documento de tesis, sin tocar
```
