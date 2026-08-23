# Estudio de curva de aprendizaje (Hipótesis específica 1)

## Por qué existe esto

La hipótesis específica 1 del TFM plantea que la caracterización y preparación
del dataset permite obtener secuencias "suficientemente consistentes" para
servir de base al entrenamiento. Los runs `debug_025/050/100/200` (versión
anterior del proyecto) no responden esa pregunta: usaban una única muestra
determinista de vehículos por tamaño (`orderBy(vehicle_id).limit(n)`, siempre
los mismos vehículos) y además reducían validation/test al mismo tamaño que
train — confundiendo dos variables distintas ("¿el modelo ve menos datos de
entrenamiento?" vs. "¿lo estoy evaluando con una muestra de test más chica y
más ruidosa?"). Con eso, cualquier diferencia entre tamaños podía deberse a
cualquiera de las dos causas, o a pura casualidad de qué vehículos cayeron en
esa única muestra — no hay forma de saberlo con un solo punto por tamaño.

## Diseño

- **Solo train se muestrea**, y de forma verdaderamente aleatoria
  (`F.rand(seed)`, no un `orderBy().limit()` determinista), únicamente entre
  vehículos normales (mismo criterio semi-supervisado que `full`).
- **Validation y test permanecen en su tamaño oficial completo** en todos los
  puntos de la curva — igual que en `full`. Así, cualquier cambio en la
  métrica solo puede atribuirse al volumen de entrenamiento.
- **Cada tamaño se repite con varias semillas** (`n_seeds`, por defecto 3),
  para poder calcular una media y una dispersión por tamaño, y distinguir una
  tendencia real de ruido de muestreo de una sola corrida.
- **El run `full` sirve de ancla** en el extremo derecho de la curva (todos
  los vehículos normales de train, sin repetición — repetirlo con varias
  semillas sería prohibitivo en tiempo de Colab). Por eso se grafica distinto
  (marcador sin banda), y no debe leerse con el mismo peso estadístico que los
  puntos con repeticiones.
- **Un solo modelo de referencia** (`learning_curve.model`, por defecto
  `lstm_autoencoder`) para mantener el costo de cómputo acotado — este estudio
  responde la hipótesis 1 (suficiencia de los datos), no la comparación entre
  arquitecturas (hipótesis 2), que ya la responde `full` con los tres modelos.

## Qué NO es

No es una búsqueda de hiperparámetros ni una comparación de arquitecturas.
No reemplaza a `full` como resultado experimental principal — es evidencia
complementaria específica para la hipótesis 1. La banda sombreada en las
figuras es ± una desviación estándar entre semillas, no un intervalo de
confianza formal; con solo 3 semillas por tamaño, hay que interpretarla como
una indicación de dispersión, no como un test estadístico riguroso.

## Cómo ejecutarlo

```python
!python scripts/run_learning_curve.py \
  --config config/config.full.yaml \
  --sizes 25,50,100,200 \
  --n-seeds 3 \
  --model lstm_autoencoder
```

Esto corre 12 combinaciones (4 tamaños × 3 semillas), cada una como su propio
`--run-name` (`learning_curve_n025_s1`, `learning_curve_n025_s2`, ...),
reutilizando el mismo flujo prepare → train → evaluate → report de siempre.
Al terminar:

```python
!python scripts/build_learning_curve_summary.py --config config/config.full.yaml
```

genera `experiments/learning_curve_summary/` con la tabla de puntos y las
figuras (`pr_auc`, `f1_score`, `recall`, `roc_auc`) — incorporando además el
run `full` ya existente como ancla, sin necesidad de recalcularlo.

## Costo esperado en Colab

Los tamaños 25-200 entrenan en segundos por modelo (el cuello de botella real
está en el arranque de Spark/EDA por combinación, no en el entrenamiento). El
costo total de las 12 combinaciones es una fracción pequeña del tiempo que
toma `full`. Si el tiempo de Colab es una restricción real, reducir
`--n-seeds` a 2 sigue siendo válido (pierde algo de precisión en la
desviación estándar, no invalida el análisis).
