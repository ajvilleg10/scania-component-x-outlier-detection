# Guía de ejecución en Google Colab

## 1. Montar Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 3. Crear carpetas

```bash
python scripts/create_drive_folders.py --config config/config.colab.yaml
```

## 4. Descargar dataset desde Kaggle y copiarlo a Drive/raw

```bash
python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml
```

## 5. Validar archivos raw

```bash
python scripts/check_raw_files.py --config config/config.colab.yaml
```

## 6. Validar configuración sin procesos pesados

```bash
python main.py --config config/config.colab.yaml --stage all --model all --mode debug --dry-run
```

## 7. Ejecución debug

```bash
python main.py --config config/config.colab.yaml --stage all --model all --mode debug
```

## 8. Ejecución final

```bash
python main.py --config config/config.colab.yaml --stage all --model all --mode full
```

## Nota importante

El pipeline corregido no usa Pandas ni `toPandas()` para procesar el dataset. Spark genera las ventanas en Parquet y PyTorch las lee por batches durante entrenamiento y evaluación.

## Nota de optimización

Si Colab se vuelve lento o aparece `ConnectionRefusedError`, `Py4JNetworkError` o caída de la JVM, normalmente significa que Spark intentó ejecutar un plan demasiado pesado para la memoria disponible. Esta versión mitiga ese problema de tres formas:

1. Filtrado temprano de vehículos en modo `debug`.
2. Preprocesamiento escalar con Spark, sin `VectorAssembler` ni `StandardScaler`.
3. Windowing Spark guardado en Parquet particionado.

Secuencia recomendada:

```bash
python main.py --config config/config.colab.yaml --stage check-data --mode debug
python main.py --config config/config.colab.yaml --stage eda --mode debug
python main.py --config config/config.colab.yaml --stage preprocess --mode debug
python main.py --config config/config.colab.yaml --stage train --model lstm_autoencoder --mode debug
python main.py --config config/config.colab.yaml --stage evaluate --model lstm_autoencoder --mode debug
```

Luego subir progresivamente `execution.max_vehicles_debug` antes de ejecutar todos los modelos.
