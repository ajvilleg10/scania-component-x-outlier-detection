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
  --model lstm_autoencoder \
  --prepare-only
```

## Entrenamiento por modelo

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --model lstm_autoencoder \
  --skip-download --skip-eda --skip-preprocess
```

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --model cnn_lstm_autoencoder \
  --skip-download --skip-eda --skip-preprocess
```

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --max-vehicles 25 \
  --model transformer_encoder \
  --skip-download --skip-eda --skip-preprocess
```

## Comparación

```bash
python scripts/run_colab_safe.py \
  --config config/config.colab.yaml \
  --mode debug \
  --compare-only
```

## Escalado

Subir gradualmente:

- 25 vehículos
- 50 vehículos
- 100 vehículos
- 200 vehículos
- full por etapas

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
