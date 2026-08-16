# Descripción del dataset

El proyecto utiliza **SCANIA Component X**, un dataset público de mantenimiento predictivo compuesto por lecturas operacionales multivariadas de componentes de motores de vehículos pesados.

## Particiones disponibles

| Partición | Archivos | Uso metodológico |
|---|---|---|
| Train | `train_operational_readouts.csv`, `train_tte.csv`, `train_specifications.csv` | Entrenamiento de modelos y construcción de la representación del comportamiento operacional frecuente. |
| Validation | `validation_operational_readouts.csv`, `validation_labels.csv`, `validation_specifications.csv` | Ajuste de umbrales, validación de hiperparámetros y selección de configuración. |
| Test | `test_operational_readouts.csv`, `test_labels.csv`, `test_specifications.csv` | Evaluación final con reporte de métricas sobre datos no vistos. |

## Consideración sobre etiquetas

Los archivos `validation_labels.csv` y `test_labels.csv` contienen la columna `class_label`, que se utiliza como referencia de evaluación a nivel de vehículo. Por esta razón, el proyecto calcula puntuaciones de atipicidad por ventana y posteriormente las agrega a nivel de vehículo para la evaluación principal.

## Especificaciones

Los archivos de especificaciones se tratan como información auxiliar. Su incorporación al modelado queda condicionada a la estabilidad del pipeline principal con lecturas operacionales.
