# SCANIA Component X · Detección de outliers temporales multivariados

Proyecto de Trabajo de Fin de Máster orientado a la **detección de outliers temporales multivariados** en lecturas operacionales de componentes de motores de vehículos pesados, dentro de un contexto de **mantenimiento predictivo**.

El repositorio está preparado para trabajar con **GitHub + Google Colab + Google Drive**. El código fuente se versiona en GitHub, mientras que los datos pesados, ventanas generadas, checkpoints y salidas experimentales se almacenan en Google Drive.

> Nota terminológica: en el código pueden aparecer nombres como `anomaly_score` por compatibilidad con la literatura técnica de *anomaly detection*. En el documento del TFM y en la documentación visible se prioriza el término **outlier** y la expresión **puntuación de atipicidad**.

## Alcance experimental

El proyecto se centra en tres modelos principales:

1. **LSTM Autoencoder**: modelo base para representar trayectorias operacionales frecuentes y detectar desviaciones mediante error de reconstrucción.
2. **CNN-LSTM Autoencoder**: modelo híbrido orientado a capturar patrones locales y dependencias temporales.
3. **Transformer Encoder simplificado**: arquitectura controlada basada en atención para estudiar relaciones temporales de mayor alcance.

Modelos como USAD, Anomaly Transformer, MTAD-GAT, OmniAnomaly y TimesNet se mantienen como soporte del estado del arte o líneas futuras, salvo que el cronograma permita una implementación adicional estable.

## Estructura del repositorio

```text
scania-component-x-outlier-detection/
├── config/                         # Configuración central del proyecto
├── notebooks/                      # Notebooks ejecutables en Colab
├── scripts/                        # Ejecuciones reproducibles por línea de comandos
├── src/scania_anomaly/             # Código reutilizable del proyecto
├── tests/                          # Pruebas mínimas de componentes críticos
├── data/                           # Marcadores; los datos reales van en Drive
├── models/                         # Marcadores; checkpoints reales van en Drive
├── reports/                        # Marcadores; resultados reales van en Drive
└── docs/                           # Documentación metodológica y operativa
```

## Estructura esperada en Google Drive

```text
/content/drive/MyDrive/TFM_SCANIA/
├── raw/
│   ├── train_operational_readouts.csv
│   ├── train_tte.csv
│   ├── train_specifications.csv
│   ├── validation_operational_readouts.csv
│   ├── validation_labels.csv
│   ├── validation_specifications.csv
│   ├── test_operational_readouts.csv
│   ├── test_labels.csv
│   └── test_specifications.csv
├── processed/
│   ├── preprocessing_metadata.json
│   ├── train_windows.npz
│   ├── validation_windows.npz
│   └── test_windows.npz
├── models/
├── outputs/
│   ├── metrics/
│   ├── tables/
│   └── figures/
└── doc/
```

También se soporta la ruta alternativa:

```text
/content/drive/MyDrive/TFM_SCANIA/data/raw/Dataset/
```

## Flujo metodológico

```text
01 EDA y calidad de datos con PySpark
   ↓
02 Preprocesamiento ajustado solo con train
   ↓
03 Construcción de ventanas temporales multivariadas
   ↓
04 Entrenamiento LSTM Autoencoder
   ↓
05 Entrenamiento CNN-LSTM Autoencoder
   ↓
06 Entrenamiento Transformer Encoder simplificado
   ↓
07 Agregación de puntuaciones por vehículo/trayectoria
   ↓
08 Comparación final sobre test
```

La regla experimental principal es:

- **Train**: ajuste de parámetros de preprocesamiento y entrenamiento de modelos.
- **Holdout interno de train**: validación de pérdida y early stopping.
- **Validation**: selección de umbral e hiperparámetros.
- **Test**: evaluación final y reporte de métricas definitivas.

## Evaluación principal

Las etiquetas disponibles en `validation_labels.csv` y `test_labels.csv` se manejan como referencias a nivel de vehículo mediante la columna `class_label`. Por ello, la evaluación principal se realiza a nivel de **vehículo/trayectoria**:

```text
Ventanas temporales → puntuación de atipicidad por ventana → agregación por vehículo → comparación con class_label
```

Las métricas por ventana se conservan como análisis exploratorio cuando resulte metodológicamente pertinente.

## Ejecución en Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/TU_USUARIO/scania-component-x-outlier-detection.git
%cd scania-component-x-outlier-detection
!pip install -e .
```

Luego ejecutar los notebooks en orden.

## Modos de ejecución

El archivo `config/config.yaml` incluye un modo de ejecución:

```yaml
execution:
  mode: debug   # debug | full
```

- `debug`: usa un subconjunto acotado de vehículos para validar el pipeline.
- `full`: ejecuta los experimentos finales con el conjunto completo disponible.

Si por restricciones computacionales se reportan resultados con una muestra, se debe documentar explícitamente el criterio de selección, número de vehículos y limitaciones.

## Buenas prácticas aplicadas

- Separación entre notebooks, código fuente, configuración, documentación y resultados.
- No versionar CSV pesados, ventanas `.npz`, checkpoints ni salidas generadas.
- Configuración centralizada en `config/config.yaml`.
- Ajuste de imputación, escalado y selección de variables solo con `train`.
- Separación entre `train`, `validation` y `test` para evitar fuga de información.
- Evaluación principal a nivel vehículo/trayectoria cuando las etiquetas estén a ese nivel.
- Registro de métricas, predicciones, metadatos y configuración por experimento.
- Pruebas mínimas para windowing, etiquetas, métricas, umbrales y agregación por vehículo.

## Nota sobre alcance

El repositorio está diseñado para evolucionar con el avance del TFM. Si el Transformer Encoder simplificado no entrega resultados estables dentro del cronograma, se recomienda priorizar una comparación rigurosa entre LSTM Autoencoder y CNN-LSTM Autoencoder, documentando el enfoque basado en atención como línea complementaria o futura.
