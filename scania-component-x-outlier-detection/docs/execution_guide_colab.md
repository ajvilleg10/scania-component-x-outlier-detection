# Guía de ejecución segura en Google Colab

Esta guía describe el flujo recomendado para ejecutar el proyecto sin saturar Colab.

## Regla principal

No entrenar los tres modelos en una misma corrida. La ejecución recomendada es:

1. Preparar datos una sola vez.
2. Entrenar y evaluar `lstm_autoencoder`.
3. Entrenar y evaluar `cnn_lstm_autoencoder`.
4. Entrenar y evaluar `transformer_encoder`.
5. Comparar resultados.

## Preparación inicial

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model lstm_autoencoder \
  --prepare-only
```

## Entrenamiento por modelo

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model lstm_autoencoder \
  --skip-download --skip-eda --skip-preprocess
```

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model cnn_lstm_autoencoder \
  --skip-download --skip-eda --skip-preprocess
```

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --run-name debug_025 \
  --model transformer_encoder \
  --skip-download --skip-eda --skip-preprocess
```

## Comparación

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_025 \
  --compare-only
```


## Archivo automático por corrida

Usar siempre `--run-name` para conservar los resultados de cada tamaño de prueba. El runner archiva los artefactos disponibles en:

```text
/content/drive/MyDrive/TFM_SCANIA/experiments/runs/<run-name>/
```

Nombres recomendados:

| Ejecución | Comando | Carpeta de archivo |
|---|---|---|
| Debug 25 | `--max-vehicles 25 --run-name debug_025` | `experiments/runs/debug_025/` |
| Debug 50 | `--max-vehicles 50 --run-name debug_050` | `experiments/runs/debug_050/` |
| Debug 100 | `--max-vehicles 100 --run-name debug_100` | `experiments/runs/debug_100/` |
| Debug 200 | `--max-vehicles 200 --run-name debug_200` | `experiments/runs/debug_200/` |
| Full | `--mode full --run-name full` | `experiments/runs/full/` |

Por defecto no se copian las ventanas Parquet para evitar duplicar archivos grandes. Para archivarlas explícitamente se puede añadir `--archive-windows`.

## Escalado

Subir gradualmente:

- `debug_025`: 25 vehículos
- `debug_050`: 50 vehículos
- `debug_100`: 100 vehículos
- `debug_200`: 200 vehículos
- `full`: todos los vehículos disponibles, por etapas

## Reanudación

Si ya existen las ventanas Parquet, usar:

```bash
--skip-download --skip-eda --skip-preprocess
```

Si solo se quiere evaluar un modelo ya entrenado:

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --model lstm_autoencoder \
  --evaluate-only
```


## Ejemplo completo para `debug_050`

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 50 \
  --run-name debug_050 \
  --model lstm_autoencoder \
  --prepare-only \
  --skip-download
```

Luego se entrenan los modelos uno a uno reutilizando el mismo `--run-name debug_050`:

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 50 \
  --run-name debug_050 \
  --model lstm_autoencoder \
  --skip-download --skip-eda --skip-preprocess
```

Finalmente:

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --run-name debug_050 \
  --compare-only
```
