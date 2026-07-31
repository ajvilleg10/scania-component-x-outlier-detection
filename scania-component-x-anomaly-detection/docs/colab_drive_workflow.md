# Flujo de trabajo con Google Colab y Google Drive

## 1. Montar Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Clonar repositorio

```python
!git clone https://github.com/TU_USUARIO/scania-component-x-outlier-detection.git
%cd scania-component-x-outlier-detection
!pip install -e .
```

## 3. Validar carpetas

```python
from scania_anomaly.config import load_config, ensure_directories
config = load_config()
ensure_directories(config)
```

## 4. Mantener GitHub limpio

No subir a GitHub:

- CSV originales;
- ventanas `.npz`;
- checkpoints `.pt` o `.pth`;
- salidas grandes;
- archivos temporales de Colab.

GitHub debe conservar código, notebooks, documentación y configuración. Los datos y resultados pesados deben permanecer en Google Drive.
