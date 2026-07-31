# Changelog TFM

## v1.0.0 - Alineación final con detección de outliers

- Se actualizó la terminología visible de "anomalías" a "outliers temporales multivariados".
- Se añadió `class_label` como columna principal de etiquetas en validation y test.
- Se cambió el manejo de vehículos sin etiqueta a `-1` para evitar asumir normalidad sin referencia.
- Se incorporó evaluación principal a nivel vehículo/trayectoria mediante agregación de puntuaciones por ventana.
- Se añadieron funciones para agregación de scores: máximo, media, percentil 95 y proporción de ventanas atípicas.
- Se reforzó el preprocesamiento ajustado solo con train para evitar fuga de información.
- Se incorporó modo `debug` / `full` en configuración.
- Se añadió análisis de irregularidad temporal mediante diferencias entre `time_step` consecutivos.
- Se mejoró la documentación metodológica con criterios de MLOps y reproducibilidad.
- Se añadieron pruebas para `class_label`, agregación a nivel vehículo y análisis temporal.
