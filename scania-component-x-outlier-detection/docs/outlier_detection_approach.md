# Enfoque de detección de outliers

El enfoque principal se basa en modelos de reconstrucción y representación temporal. La idea metodológica es que un modelo entrenado sobre patrones frecuentes de operación tienda a reconstruir mejor trayectorias normales que trayectorias atípicas. La diferencia entre la entrada y la reconstrucción se interpreta como **puntuación de atipicidad** u **outlier score**.

## Modelos principales

1. **LSTM Autoencoder**: aprendizaje de dependencias temporales mediante reconstrucción secuencial.
2. **CNN-LSTM Autoencoder**: extracción de patrones locales y posterior modelado temporal.
3. **Transformer Encoder simplificado**: mecanismo de atención controlado para capturar dependencias de mayor alcance.

## Umbral de decisión

El umbral se ajusta sobre `validation`, nunca sobre `test`. La estrategia base puede ser el percentil 95 de los scores de validation. Cuando las etiquetas disponibles lo permitan, se puede seleccionar el umbral que maximice F1-score en validation.

## Nivel de evaluación

Las etiquetas `class_label` de validation y test se manejan como referencias a nivel de vehículo. Por ello, la evaluación principal se realiza agregando las puntuaciones de ventana a nivel vehículo o trayectoria. Las métricas por ventana pueden utilizarse como análisis complementario, pero no deben sustituir la evaluación principal si la referencia disponible no está definida a nivel de ventana.
