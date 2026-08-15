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
