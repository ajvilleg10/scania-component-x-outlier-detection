# Arquitectura del proyecto

## Separación de responsabilidades

El repositorio Git contiene únicamente código, configuración, pruebas, notebooks de apoyo y documentación técnica. Google Drive contiene los CSV originales, la caché de ventanas Parquet y los artefactos de cada experimento.

```text
Git repo                                Google Drive
------------------------------          ---------------------------------------
main.py                                 TFM_SCANIA/data/raw
config/                                 TFM_SCANIA/data/processed/windows
scripts/                                TFM_SCANIA/data/processed/metadata
src/                                    TFM_SCANIA/data/processed/manifests
tests/                                  TFM_SCANIA/experiments/runs/<run-name>
notebooks/
docs/
```

Cada `run-name` es autocontenido respecto de los resultados: modelos, métricas, predicciones, comparaciones, tablas, figuras, logs, manifiestos y configuración efectiva. Las ventanas no se duplican por defecto porque pueden ocupar mucho espacio; se tratan como una caché reproducible derivada de los CSV raw.

## Flujo

1. `--prepare-only`: validación de raw, EDA, preprocesamiento y windowing.
2. Ejecución individual de cada modelo: entrenamiento + evaluación.
3. `--compare-only`: consolidación de métricas de los tres modelos del run.
4. `--report-only`: inventario y resumen del experimento.

El proyecto valida que las ventanas activas correspondan al mismo `run-name` antes de entrenar o evaluar.


## Síntesis transversal de experimentos

Cada run es autosuficiente y no sobrescribe a los demás. Al terminar el estudio, `--study-summary` lee únicamente las métricas persistidas en `experiments/runs/*/metrics` y crea `experiments/study_summary/`. Esta carpeta contiene la comparación longitudinal de los runs debug y full, sin modificar los artefactos originales.
