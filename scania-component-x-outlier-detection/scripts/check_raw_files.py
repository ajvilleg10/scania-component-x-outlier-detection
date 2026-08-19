from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import argparse

from scania_outliers.config import get_dataset_files, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida que los CSV requeridos estén en Google Drive/data/raw.")
    parser.add_argument("--config", default="config/config.colab.yaml", help="Archivo YAML de configuración.")
    parser.add_argument(
        "--allow-alternative",
        action="store_true",
        help="Permite aceptar archivos en raw_dir_alternative. Por defecto se exige paths.raw_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_dir"])
    alt_raw_dir = Path(config["paths"].get("raw_dir_alternative", ""))
    files = get_dataset_files(config)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio raw esperado: {raw_dir}\n"
            "Primero monte Google Drive y ejecute scripts/create_drive_folders.py."
        )

    print(f"Validando archivos obligatorios en: {raw_dir}")
    if args.allow_alternative:
        print(f"También se acepta ruta alternativa: {alt_raw_dir}")

    missing = []
    found = []
    found_in_alt = []

    for alias, filename in files.items():
        primary = raw_dir / filename
        alternative = alt_raw_dir / filename if alt_raw_dir else None
        if primary.exists():
            found.append((alias, filename, primary))
            print(f"OK: {alias} -> {filename}")
        elif args.allow_alternative and alternative and alternative.exists():
            found_in_alt.append((alias, filename, alternative))
            print(f"OK alternativo: {alias} -> {filename} en {alternative}")
        else:
            missing.append((alias, filename))
            print(f"FALTA: {alias} -> {filename}")

    if missing:
        missing_text = "\n".join(f"- {alias}: {filename}" for alias, filename in missing)
        raise FileNotFoundError(
            "Faltan archivos obligatorios en data/raw:\n"
            f"{missing_text}\n\n"
            "Ejecute primero:\n"
            "python scripts/download_kaggle_to_drive.py --config config/config.colab.yaml\n\n"
            f"Destino esperado: {raw_dir}"
        )

    print("\nValidación completada.")
    print(f"Archivos encontrados en raw_dir: {len(found)}")
    if found_in_alt:
        print(f"Archivos encontrados en ruta alternativa: {len(found_in_alt)}")
    print("Todos los archivos requeridos están listos para main.py.")


if __name__ == "__main__":
    main()
