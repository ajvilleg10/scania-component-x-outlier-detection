from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from scania_outliers.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea únicamente las carpetas compartidas necesarias en Google Drive.")
    parser.add_argument("--config", default="config/config.colab.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    p = config.get("paths", {})
    processed = Path(p.get("processed_dir", "data/processed"))
    experiments = Path(p.get("experiments_dir", "experiments"))

    folders = [
        Path(p.get("drive_root", ".")),
        Path(p.get("raw_dir", "data/raw")),
        processed,
        processed / "windows",
        processed / "metadata",
        processed / "manifests",
        experiments / "runs",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print("Created/checked:", folder)

    print("\nLas carpetas de modelos, métricas, predicciones, tablas, figuras y logs se crean bajo experiments/runs/<run-name> solo cuando se usan.")


if __name__ == "__main__":
    main()
