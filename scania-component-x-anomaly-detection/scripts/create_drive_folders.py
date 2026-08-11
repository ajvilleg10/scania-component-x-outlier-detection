from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import argparse
from pathlib import Path

from scania_outliers.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea la estructura de carpetas esperada en Google Drive.")
    parser.add_argument("--config", default="config/config.colab.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})

    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    models_dir = Path(paths.get("models_dir", "models"))
    experiments_dir = Path(paths.get("experiments_dir", outputs_dir / "experiments"))
    doc_dir = Path(paths.get("doc_dir", "doc"))

    folders = [
        Path(paths.get("drive_root", ".")),
        Path(paths.get("raw_dir", "data/raw")),
        processed_dir,
        processed_dir / "quality_reports",
        processed_dir / "clean",
        processed_dir / "windows",
        processed_dir / "metadata",
        processed_dir / "manifests",
        models_dir,
        models_dir / "lstm_autoencoder",
        models_dir / "cnn_lstm_autoencoder",
        models_dir / "transformer_encoder_simplified",
        outputs_dir,
        Path(paths.get("figures_dir", outputs_dir / "figures")),
        Path(paths.get("tables_dir", outputs_dir / "tables")),
        Path(paths.get("metrics_dir", outputs_dir / "metrics")),
        outputs_dir / "predictions",
        outputs_dir / "comparisons",
        outputs_dir / "logs",
        outputs_dir / "reports",
        outputs_dir / "runs",
        experiments_dir,
        experiments_dir / "runs",
        experiments_dir / "registry",
        doc_dir,
        doc_dir / "metodologia",
        doc_dir / "resultados",
        doc_dir / "anexos",
        doc_dir / "defensa",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print("Created/checked:", folder)


if __name__ == "__main__":
    main()
