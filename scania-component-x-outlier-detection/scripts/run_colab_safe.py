from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from scania_outliers.config import load_config  # noqa: E402

VALID_MODELS = ["lstm_autoencoder", "cnn_lstm_autoencoder", "transformer_encoder"]


def sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise ValueError("--run-name no puede quedar vacío.")
    return cleaned


def default_run_name(mode: str, max_vehicles: int) -> str:
    return f"debug_{max_vehicles:03d}" if mode == "debug" else "full"


def run_command(command: list[str], log_file: Path) -> None:
    print("\n" + "=" * 100)
    print("Ejecutando:", " ".join(command))
    print("Log:", log_file)
    print("=" * 100 + "\n")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_file.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(command) + "\n")
        f.write("START: " + datetime.now().isoformat(timespec="seconds") + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        code = process.wait()
        elapsed = round((time.time() - start) / 60, 3)
        f.write(f"\nEND: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"ELAPSED_MINUTES: {elapsed}\nRETURN_CODE: {code}\n")
    if code != 0:
        raise SystemExit(f"Falló una etapa. Revise el log: {log_file}")


def raw_files_complete(config: dict[str, Any]) -> bool:
    raw_dir = Path(config["paths"]["raw_dir"])
    files = config.get("dataset", {}).get("files", {})
    return raw_dir.exists() and all((raw_dir / filename).exists() for filename in files.values())


def main_command(py: str, config_path: Path, run_name: str, mode: str, max_vehicles: int, stage: str, model: str | None = None) -> list[str]:
    command = [
        py,
        "main.py",
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--mode",
        mode,
        "--run-id",
        run_name,
    ]
    if mode == "debug":
        command += ["--max-vehicles", str(max_vehicles)]
    if model is not None:
        command += ["--model", model]
    return command


def save_runner_event(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "runner_events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Runner profesional para Colab: prepare una vez, entrene/evalúe un modelo por comando, "
            "compare al final y conserve cada experimento mediante --run-name."
        )
    )
    parser.add_argument("--config", default="config/config.colab.yaml")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--max-vehicles", type=int, default=None, help="Solo aplica a debug. Si se omite y run-name es debug_050, se infiere 50.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", choices=VALID_MODELS, default=None)
    parser.add_argument("--force-download", action="store_true", help="Fuerza la descarga/copia desde KaggleHub durante prepare-only.")

    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true", help="Datos, EDA, preprocesamiento y ventanas. No requiere --model.")
    phase.add_argument("--train-only", action="store_true", help="Solo entrenamiento. Requiere --model y preparación previa del mismo run.")
    phase.add_argument("--evaluate-only", action="store_true", help="Solo evaluación. Requiere --model previamente entrenado.")
    phase.add_argument("--compare-only", action="store_true", help="Consolida las métricas del run actual.")
    phase.add_argument("--report-only", action="store_true", help="Genera inventario/resumen del run y verifica las figuras existentes.")
    phase.add_argument("--study-summary", action="store_true", help="Consolida métricas y figuras entre todos los runs disponibles.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    configured_max = int(config.get("execution", {}).get("max_vehicles_debug") or 25)
    requested_name = sanitize_run_name(args.run_name) if args.run_name else None
    inferred_from_name = None
    if requested_name and args.mode == "debug":
        match = re.fullmatch(r"debug_(\d+)", requested_name)
        if match:
            inferred_from_name = int(match.group(1))
    max_vehicles = int(args.max_vehicles if args.max_vehicles is not None else (inferred_from_name or configured_max))
    run_name = requested_name or sanitize_run_name(default_run_name(args.mode, max_vehicles))
    if args.mode == "debug" and inferred_from_name is not None and inferred_from_name != max_vehicles:
        raise SystemExit(f"Inconsistencia: --run-name {run_name} implica {inferred_from_name} vehículos, pero --max-vehicles={max_vehicles}.")

    experiments_dir = Path(config.get("paths", {}).get("experiments_dir", Path(config["paths"]["drive_root"]) / "experiments"))
    run_dir = experiments_dir / "runs" / run_name
    logs_dir = run_dir / "logs"
    py = sys.executable

    if args.study_summary:
        summary_dir = experiments_dir / "study_summary"
        run_command(
            [py, "scripts/build_study_summary.py", "--config", str(config_path)],
            summary_dir / "logs" / "build_study_summary.log",
        )
        return 0

    if not any([args.prepare_only, args.compare_only, args.report_only, args.study_summary]) and args.model is None:
        raise SystemExit("Indique --model para entrenar/evaluar, o use --prepare-only/--compare-only/--report-only/--study-summary.")

    print("\nPLAN DE EJECUCIÓN")
    print("Run:", run_name)
    print("Modo:", args.mode)
    print("Vehículos debug:", max_vehicles if args.mode == "debug" else "todos")
    print("Modelo:", args.model or "no aplica")
    print("Run directory:", run_dir)

    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "max_vehicles": max_vehicles if args.mode == "debug" else None,
        "model": args.model,
        "prepare_only": args.prepare_only,
        "train_only": args.train_only,
        "evaluate_only": args.evaluate_only,
        "compare_only": args.compare_only,
        "report_only": args.report_only,
    }
    save_runner_event(run_dir, event)

    if args.prepare_only:
        run_command([py, "scripts/create_drive_folders.py", "--config", str(config_path)], logs_dir / "01_create_drive_folders.log")
        if args.force_download or not raw_files_complete(config):
            download_cmd = [py, "scripts/download_kaggle_to_drive.py", "--config", str(config_path)]
            if args.force_download:
                download_cmd.append("--force")
            run_command(download_cmd, logs_dir / "02_download_kaggle_to_drive.log")
        else:
            print("Los 9 archivos raw ya existen en Drive; se reutilizan sin volver a descargar.")
        run_command([py, "scripts/check_raw_files.py", "--config", str(config_path)], logs_dir / "03_check_raw_files.log")
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "check-data"), logs_dir / "04_check_data.log")
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "eda"), logs_dir / "05_eda.log")
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "preprocess"), logs_dir / "06_preprocess_windowing.log")
        print("\nPreparación completada. Ahora ejecute cada modelo con el MISMO --run-name.")
        return 0

    if args.compare_only:
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "compare"), logs_dir / "90_compare.log")
        return 0

    if args.report_only:
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "report"), logs_dir / "99_report.log")
        return 0

    assert args.model is not None
    if args.train_only:
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "train", args.model), logs_dir / f"10_train_{args.model}.log")
        return 0

    if args.evaluate_only:
        run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "evaluate", args.model), logs_dir / f"20_evaluate_{args.model}.log")
        return 0

    # Normal model command = train + evaluate. No skip flags are needed.
    run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "train", args.model), logs_dir / f"10_train_{args.model}.log")
    run_command(main_command(py, config_path, run_name, args.mode, max_vehicles, "evaluate", args.model), logs_dir / f"20_evaluate_{args.model}.log")
    print(f"\nModelo {args.model} entrenado y evaluado para run {run_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
