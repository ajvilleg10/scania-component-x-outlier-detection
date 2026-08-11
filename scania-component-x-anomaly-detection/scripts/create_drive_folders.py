from __future__ import annotations

import argparse
from pathlib import Path

from scania_outliers.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Crea la estructura de carpetas esperada en Google Drive.")
    parser.add_argument("--config", default="config/config.colab.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})
    folders = [
        paths.get("raw_dir"),
        paths.get("processed_dir"),
        paths.get("models_dir"),
        paths.get("outputs_dir"),
        paths.get("figures_dir"),
        paths.get("tables_dir"),
        paths.get("metrics_dir"),
        paths.get("doc_dir"),
        str(Path(paths.get("processed_dir", "data/processed")) / "quality_reports"),
        str(Path(paths.get("processed_dir", "data/processed")) / "clean"),
        str(Path(paths.get("processed_dir", "data/processed")) / "windows"),
        str(Path(paths.get("processed_dir", "data/processed")) / "metadata"),
        str(Path(paths.get("processed_dir", "data/processed")) / "manifests"),
        str(Path(paths.get("outputs_dir", "outputs")) / "predictions"),
        str(Path(paths.get("outputs_dir", "outputs")) / "comparisons"),
        str(Path(paths.get("outputs_dir", "outputs")) / "logs"),
        str(Path(paths.get("outputs_dir", "outputs")) / "reports"),
    ]
    for folder in folders:
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            print("Created/checked:", folder)


if __name__ == "__main__":
    main()
