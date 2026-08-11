from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scania_outliers.config import load_config
from scania_outliers.pipelines.orchestrator import ScaniaOutlierPipeline

VALID_STAGES = ["all", "check-data", "eda", "preprocess", "train", "evaluate", "compare"]
VALID_MODELS = ["all", "lstm_autoencoder", "cnn_lstm_autoencoder", "transformer_encoder_simplified"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scania-outliers",
        description="Pipeline automatizado para detección de outliers temporales multivariados en SCANIA Component X.",
    )
    parser.add_argument("--config", default="config/config.colab.yaml", help="Ruta del archivo YAML de configuración.")
    parser.add_argument("--stage", choices=VALID_STAGES, default="all", help="Etapa del pipeline a ejecutar.")
    parser.add_argument("--model", choices=VALID_MODELS, default="all", help="Modelo a entrenar o evaluar.")
    parser.add_argument("--mode", choices=["debug", "full"], default=None, help="Sobrescribe execution.mode del YAML.")
    parser.add_argument("--run-id", default=None, help="Identificador opcional de la ejecución.")
    parser.add_argument("--dry-run", action="store_true", help="Valida configuración y muestra el plan sin ejecutar etapas pesadas.")
    parser.add_argument("--no-spark-stop", action="store_true", help="No detener Spark al finalizar.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.mode is not None:
        config.setdefault("execution", {})["mode"] = args.mode

    pipeline = ScaniaOutlierPipeline(config=config, config_path=Path(args.config), run_id=args.run_id)

    if args.dry_run:
        pipeline.dry_run(stage=args.stage, model=args.model)
        return 0

    try:
        pipeline.run(stage=args.stage, model=args.model)
    finally:
        if not args.no_spark_stop:
            pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
