# Metodología experimental

El proyecto sigue un flujo reproducible de detección de anomalías en series temporales multivariadas:

1. Análisis exploratorio del dataset con PySpark.
2. Limpieza y selección de variables operacionales.
3. Integración de etiquetas disponibles y registros de reparación.
4. Normalización ajustada sobre train y aplicada a validation/test.
5. Construcción de ventanas temporales por vehículo.
6. Entrenamiento de modelos de reconstrucción sobre train.
7. Selección del umbral de anomalía sobre validation.
8. Evaluación final sobre test.
9. Comparación de modelos mediante Precision, Recall, F1-score, ROC-AUC y PR-AUC.
10. Discusión de resultados desde el contexto de mantenimiento predictivo.

## Regla de evaluación

El conjunto de test no se usa para seleccionar hiperparámetros ni umbrales. Solo se emplea para el reporte final de métricas.

## Nivel de agregación

El análisis principal se realiza a nivel de ventana temporal. Cuando sea necesario para la discusión de mantenimiento predictivo, los scores se agregan a nivel de vehículo mediante máximo, media y proporción de ventanas anómalas.
