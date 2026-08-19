# Guía de ejecución en Google Colab

## Preparación del entorno

Active GPU, instale Java 17, monte Google Drive, clone el repositorio en `/content` e instale `requirements.txt`.

## Un experimento debug

```bash
python scripts/run_colab_safe.py --config config/config.colab.yaml --mode debug --run-name debug_025 --prepare-only
python scripts/run_colab_safe.py --config config/config.colab.yaml --mode debug --run-name debug_025 --model lstm_autoencoder
python scripts/run_colab_safe.py --config config/config.colab.yaml --mode debug --run-name debug_025 --model cnn_lstm_autoencoder
python scripts/run_colab_safe.py --config config/config.colab.yaml --mode debug --run-name debug_025 --model transformer_encoder
python scripts/run_colab_safe.py --config config/config.colab.yaml --mode debug --run-name debug_025 --compare-only
python scripts/run_colab_safe.py --config config/config.colab.yaml --mode debug --run-name debug_025 --report-only
```

No se necesitan flags `skip-*`. `prepare-only` es independiente del modelo y se ejecuta una sola vez por `run-name`.

## Secuencia experimental

Repita el flujo para `debug_050`, `debug_100`, `debug_200` y finalmente `full`. No cambie de run sin volver a ejecutar `--prepare-only`, porque la caché de ventanas compartida se valida contra el run activo.


## Consolidación final del estudio

Cuando estén terminados `debug_025`, `debug_050`, `debug_100`, `debug_200` y `full`:

```bash
python scripts/run_colab_safe.py --config config/config.full.yaml --study-summary
```

Esto genera `experiments/study_summary/` con tablas y figuras de evolución entre runs.
