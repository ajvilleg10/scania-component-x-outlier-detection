from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scania_outliers.config import load_config
from scania_outliers.pipelines.orchestrator import ScaniaOutlierPipeline

VALID_STAGES = ["all", "check-data", "eda", "preprocess", "train", "evaluate", "compare", "report"]
VALID_MODELS = [
    "lstm_autoencoder",
    "cnn_lstm_autoencoder",
    "transformer_encoder",
    "transformer_encoder_simplified",
    "all",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scania-outliers",
        description="Pipeline reproducible para detección de outliers temporales multivariados en SCANIA Component X.",
    )
    parser.add_argument("--config", default="config/config.colab.yaml")
    parser.add_argument("--stage", choices=VALID_STAGES, default="check-data")
    parser.add_argument("--model", choices=VALID_MODELS, default="lstm_autoencoder")
    parser.add_argument("--mode", choices=["debug", "full"], default=None)
    parser.add_argument("--max-vehicles", type=int, default=None, help="Sobrescribe execution.max_vehicles_debug en modo debug.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-spark-stop", action="store_true")
    parser.add_argument("--allow-all-models", action="store_true")
    return parser


def _validate_safe_model_execution(args: argparse.Namespace) -> None:
    heavy_stage = args.stage in {"all", "train", "evaluate"}
    if heavy_stage and args.model == "all" and not args.allow_all_models:
        raise SystemExit(
            "Por estabilidad en Colab no se permite --model all en etapas pesadas. "
            "Ejecute un modelo por corrida y use --stage compare al final."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_safe_model_execution(args)

    config = load_config(args.config)
    if args.mode is not None:
        config.setdefault("execution", {})["mode"] = args.mode
    if config.get("execution", {}).get("mode") == "debug" and args.max_vehicles is not None:
        config.setdefault("execution", {})["max_vehicles_debug"] = int(args.max_vehicles)
    if config.get("execution", {}).get("mode") == "full":
        config.setdefault("execution", {})["max_vehicles_debug"] = None

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
