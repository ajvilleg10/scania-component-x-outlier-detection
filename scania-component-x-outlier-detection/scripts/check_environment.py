from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import argparse
from pathlib import Path

from scania_outliers.config import ensure_directories, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida entorno y rutas del proyecto SCANIA outliers.")
    parser.add_argument("--config", default="config/config.colab.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)

    print("Project:", config["project"]["name"])
    print("Objective:", config["project"].get("objective"))
    print("Execution mode:", config.get("execution", {}).get("mode"))
    print("Drive root:", config["paths"]["drive_root"])
    print("Kaggle dataset:", config.get("dataset", {}).get("kaggle", {}).get("dataset_slug"))

    raw_dir = Path(config["paths"]["raw_dir"])
    alt_raw = Path(config["paths"].get("raw_dir_alternative", ""))
    print("Raw dir exists:", raw_dir.exists(), raw_dir)
    print("Alternative raw dir exists:", alt_raw.exists(), alt_raw)

    print("\nConfigured dataset files:")
    for alias, filename in config["dataset"]["files"].items():
        exists_primary = (raw_dir / filename).exists()
        exists_alt = (alt_raw / filename).exists() if alt_raw else False
        status = "OK" if exists_primary else ("ALT" if exists_alt else "MISSING")
        print(f"- {alias}: {filename} [{status}]")

    print("\nPara validar de forma estricta los datos en Drive/data/raw ejecute:")
    print("python scripts/check_raw_files.py --config", args.config)


if __name__ == "__main__":
    main()
