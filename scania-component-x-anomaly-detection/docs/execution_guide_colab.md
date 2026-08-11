# Guía de ejecución en Google Colab

## 1. Activar GPU

Entorno de ejecución -> Cambiar tipo de entorno de ejecución -> GPU.

```bash
!nvidia-smi
```

## 2. Montar Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Ir al proyecto

```bash
%cd /content/drive/MyDrive/TFM_SCANIA/project/scania-component-x-outlier-detection-final-colab
```

## 4. Instalar dependencias

```bash
!pip install -r requirements.txt
!pip install -e .
```

## 5. Validar entorno

```bash
!python scripts/check_environment.py --config config/config.colab.yaml
```

## 6. Ejecutar debug

```bash
!python main.py --config config/config.colab.yaml --stage all --model all --mode debug
```

## 7. Ejecutar resultados finales

```bash
!python main.py --config config/config.colab.yaml --stage all --model all --mode full
```

Si Colab se desconecta, ejecutar por etapas como se indica en el README.
