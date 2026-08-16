from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import argparse
import shutil
from pathlib import Path

from scania_outliers.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga el dataset SCANIA Component X desde KaggleHub y copia los CSV a Google Drive/data/raw."
    )
    parser.add_argument("--config", default="config/config.colab.yaml", help="Archivo YAML de configuración.")
    parser.add_argument("--dataset", default=None, help="Slug de Kaggle, por ejemplo tapanbatla/scania-component-x-dataset-2025.")
    parser.add_argument("--raw-dir", default=None, help="Directorio destino data/raw en Google Drive.")
    parser.add_argument("--force", action="store_true", help="Sobrescribe archivos existentes en raw_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_slug = args.dataset or config.get("dataset", {}).get("kaggle", {}).get(
        "dataset_slug", "tapanbatla/scania-component-x-dataset-2025"
    )
    raw_dir = Path(args.raw_dir or config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "No se encontró kagglehub. Instale dependencias con: pip install -r requirements.txt"
        ) from exc

    print(f"Descargando dataset desde KaggleHub: {dataset_slug}")
    kaggle_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"Dataset descargado en caché temporal de Colab: {kaggle_path}")
    print(f"Destino persistente del proyecto: {raw_dir}")

    csv_files = sorted(kaggle_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV dentro de {kaggle_path}")

    copied = []
    skipped = []
    for csv_file in csv_files:
        dest = raw_dir / csv_file.name
        if dest.exists() and not args.force:
            skipped.append(dest.name)
            print(f"Ya existe, se conserva: {dest.name}")
            continue
        shutil.copy2(csv_file, dest)
        copied.append(dest.name)
        print(f"Copiado: {csv_file.name} -> {dest}")

    print("\nResumen de descarga/copias")
    print(f"CSV encontrados: {len(csv_files)}")
    print(f"CSV copiados: {len(copied)}")
    print(f"CSV conservados porque ya existían: {len(skipped)}")
    print("\nEl pipeline leerá exclusivamente desde data/raw en Google Drive.")


if __name__ == "__main__":
    main()
