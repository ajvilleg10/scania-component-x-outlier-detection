from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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


def run_command(command: list[str], log_file: Path, stop_on_error: bool = True) -> int:
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
        f.write(f"ELAPSED_MINUTES: {elapsed}\n")
        f.write(f"RETURN_CODE: {code}\n")

    print(f"\nFinalizado en {elapsed} min | código={code}")
    if code != 0 and stop_on_error:
        raise SystemExit(
            f"Falló una etapa. Revise el log:\n{log_file}\n\n"
            "Sugerencia: reduzca --max-vehicles o reanude saltando etapas ya completadas."
        )
    return code


def update_debug_config(config_path: Path, max_vehicles: int) -> None:
    text = config_path.read_text(encoding="utf-8")
    text = re.sub(r"mode:\s*(debug|full)", "mode: debug", text)
    text = re.sub(r"max_vehicles_debug:\s*\d+", f"max_vehicles_debug: {max_vehicles}", text)
    config_path.write_text(text, encoding="utf-8")
    print(f"Config actualizada en modo debug con max_vehicles_debug={max_vehicles}")


def sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("--run-name no puede quedar vacío después de limpiarlo.")
    return cleaned


def default_run_name(mode: str, max_vehicles: int) -> str:
    if mode == "debug":
        return f"debug_{max_vehicles:03d}"
    return "full"


def build_log_dir(config: dict[str, Any], run_name: str) -> Path:
    outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))
    return outputs_dir / "logs" / "colab_safe_runner" / run_name


def with_run_id(command: list[str], run_name: str) -> list[str]:
    """Attach the same run identifier to main.py commands so stage manifests are traceable."""
    if len(command) >= 2 and Path(command[1]).name == "main.py" and "--run-id" not in command:
        return command + ["--run-id", run_name]
    return command


def copy_tree_if_exists(source: Path, destination: Path, overwrite: bool = True) -> bool:
    if not source.exists():
        return False
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return True


def copy_file_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists() or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def archive_run_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_name: str,
    logs_dir: Path,
    mode: str,
    max_vehicles: int,
    model: str,
    flags: dict[str, bool],
    archive_windows: bool = False,
) -> Path:
    """Create a persistent snapshot under experiments/runs/<run_name>.

    The latest outputs in outputs/ are still kept for convenience, but this function
    protects each experiment from being overwritten by later runs.
    """
    paths = config.get("paths", {})
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    models_dir = Path(paths.get("models_dir", "models"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    experiments_dir = Path(paths.get("experiments_dir", outputs_dir / "experiments"))

    run_dir = experiments_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    tree_sources = {
        "metrics": outputs_dir / "metrics",
        "comparisons": outputs_dir / "comparisons",
        "predictions": outputs_dir / "predictions",
        "figures": outputs_dir / "figures",
        "tables": outputs_dir / "tables",
    }

    for label, source in tree_sources.items():
        if copy_tree_if_exists(source, run_dir / label):
            copied.append(label)

    # Copy only the logs for this run, not the complete logs directory.
    if copy_tree_if_exists(logs_dir, run_dir / "logs"):
        copied.append("logs")

    # Copy the model associated with this execution when it exists. This avoids
    # copying all model folders unnecessarily during each model run.
    model_source = models_dir / model
    if model in VALID_MODELS and copy_tree_if_exists(model_source, run_dir / "models" / model):
        copied.append(f"models/{model}")

    # Keep lightweight preprocessing metadata/manifests for reproducibility.
    for folder_name in ["metadata", "manifests"]:
        source = processed_dir / folder_name
        if copy_tree_if_exists(source, run_dir / "processed" / folder_name):
            copied.append(f"processed/{folder_name}")

    # Windows can be large; archive only when explicitly requested.
    if archive_windows:
        source = processed_dir / "windows"
        if copy_tree_if_exists(source, run_dir / "processed" / "windows"):
            copied.append("processed/windows")

    copy_file_if_exists(config_path, run_dir / "config_used.yaml")

    metadata = {
        "run_name": run_name,
        "mode": mode,
        "max_vehicles": max_vehicles if mode == "debug" else None,
        "model": model,
        "flags": flags,
        "archive_windows": archive_windows,
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "outputs_dir": str(outputs_dir),
        "logs_dir": str(logs_dir),
        "copied_artifacts": copied,
        "note": (
            "Esta carpeta conserva una copia de los resultados de la ejecución para evitar "
            "que se pierdan o se mezclen al ejecutar nuevos debug/full runs."
        ),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "model": model,
        "flags": flags,
        "copied_artifacts": copied,
    }
    with (run_dir / "run_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print("\nArchivo de resultados actualizado:")
    print(run_dir)
    if copied:
        print("Artefactos copiados:", ", ".join(copied))
    else:
        print("No se encontraron artefactos todavía; se guardó metadata/config/logs si existían.")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecución segura por etapas para Google Colab. Preparar datos una vez y entrenar un modelo por corrida."
    )
    parser.add_argument("--config", default="config/config.colab.yaml")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--max-vehicles", type=int, default=25, help="Solo aplica en modo debug.")
    parser.add_argument("--model", choices=VALID_MODELS, default="lstm_autoencoder", help="Modelo único a entrenar/evaluar.")
    parser.add_argument("--prepare-only", action="store_true", help="Solo descarga/valida/EDA/preprocesa/windowing; no entrena.")
    parser.add_argument("--train-only", action="store_true", help="Ejecuta solo entrenamiento del modelo indicado.")
    parser.add_argument("--evaluate-only", action="store_true", help="Ejecuta solo evaluación del modelo indicado.")
    parser.add_argument("--compare-only", action="store_true", help="Ejecuta solo comparación final.")
    parser.add_argument("--skip-download", action="store_true", help="No descarga dataset si ya está en Drive/data/raw.")
    parser.add_argument("--skip-eda", action="store_true", help="Omite EDA si ya fue ejecutado.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Omite preprocesamiento/windowing si las ventanas Parquet ya existen.")
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Nombre del experimento. Guarda logs y una copia de resultados en "
            "experiments/runs/<run-name>. Si no se indica, usa debug_025, debug_050, etc. o full."
        ),
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="No copia resultados a experiments/runs/<run-name>. No recomendado para corridas del TFM.",
    )
    parser.add_argument(
        "--archive-windows",
        action="store_true",
        help="También copia data/processed/windows al run. Puede ocupar mucho espacio en Drive.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.mode == "debug":
        update_debug_config(config_path, args.max_vehicles)

    run_name = sanitize_run_name(args.run_name or default_run_name(args.mode, args.max_vehicles))
    config = load_config(config_path)
    logs_dir = build_log_dir(config, run_name)
    logs_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "prepare_only": args.prepare_only,
        "train_only": args.train_only,
        "evaluate_only": args.evaluate_only,
        "compare_only": args.compare_only,
        "skip_download": args.skip_download,
        "skip_eda": args.skip_eda,
        "skip_preprocess": args.skip_preprocess,
        "no_archive": args.no_archive,
        "archive_windows": args.archive_windows,
    }

    print("\nPLAN DE EJECUCIÓN SEGURA")
    print("Config:", config_path)
    print("Run name:", run_name)
    print("Modo:", args.mode)
    print("Max vehículos:", args.max_vehicles if args.mode == "debug" else "N/A")
    print("Modelo:", args.model)
    print("Preparar solamente:", args.prepare_only)
    print("Logs:", logs_dir)
    print("Archivo de run:", Path(config.get("paths", {}).get("experiments_dir", "experiments")) / "runs" / run_name)
    print("\nNota: este runner entrena un único modelo por ejecución para reducir caídas en Colab.")

    py = sys.executable

    def cmd(command: list[str]) -> list[str]:
        return with_run_id(command, run_name)

    if args.compare_only:
        run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "compare", "--model", "all", "--mode", args.mode]), logs_dir / "99_compare.log")
        if not args.no_archive:
            archive_run_outputs(config=config, config_path=config_path, run_name=run_name, logs_dir=logs_dir, mode=args.mode, max_vehicles=args.max_vehicles, model=args.model, flags=flags, archive_windows=args.archive_windows)
        return 0

    if args.train_only:
        run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "train", "--model", args.model, "--mode", args.mode]), logs_dir / f"07_train_{args.model}.log")
        if not args.no_archive:
            archive_run_outputs(config=config, config_path=config_path, run_name=run_name, logs_dir=logs_dir, mode=args.mode, max_vehicles=args.max_vehicles, model=args.model, flags=flags, archive_windows=args.archive_windows)
        return 0

    if args.evaluate_only:
        run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "evaluate", "--model", args.model, "--mode", args.mode]), logs_dir / f"08_evaluate_{args.model}.log")
        if not args.no_archive:
            archive_run_outputs(config=config, config_path=config_path, run_name=run_name, logs_dir=logs_dir, mode=args.mode, max_vehicles=args.max_vehicles, model=args.model, flags=flags, archive_windows=args.archive_windows)
        return 0

    run_command([py, "scripts/create_drive_folders.py", "--config", str(config_path)], logs_dir / "01_create_drive_folders.log")

    if not args.skip_download:
        run_command([py, "scripts/download_kaggle_to_drive.py", "--config", str(config_path)], logs_dir / "02_download_kaggle_to_drive.log")

    run_command([py, "scripts/check_raw_files.py", "--config", str(config_path)], logs_dir / "03_check_raw_files.log")
    run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "check-data", "--mode", args.mode]), logs_dir / "04_check_data_stage.log")

    if not args.skip_eda:
        run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "eda", "--mode", args.mode]), logs_dir / "05_eda.log")

    if not args.skip_preprocess:
        run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "preprocess", "--mode", args.mode]), logs_dir / "06_preprocess_windowing.log")

    if args.prepare_only:
        if not args.no_archive:
            archive_run_outputs(config=config, config_path=config_path, run_name=run_name, logs_dir=logs_dir, mode=args.mode, max_vehicles=args.max_vehicles, model=args.model, flags=flags, archive_windows=args.archive_windows)
        print("\nPreparación terminada. Ventanas Parquet listas para entrenar modelos por separado.")
        print("Siguiente paso ejemplo:")
        print(f"python scripts/run_colab_safe.py --config {config_path} --mode {args.mode} --max-vehicles {args.max_vehicles} --run-name {run_name} --model {args.model} --skip-download --skip-eda --skip-preprocess")
        return 0

    run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "train", "--model", args.model, "--mode", args.mode]), logs_dir / f"07_train_{args.model}.log")
    run_command(cmd([py, "main.py", "--config", str(config_path), "--stage", "evaluate", "--model", args.model, "--mode", args.mode]), logs_dir / f"08_evaluate_{args.model}.log")

    if not args.no_archive:
        archive_run_outputs(config=config, config_path=config_path, run_name=run_name, logs_dir=logs_dir, mode=args.mode, max_vehicles=args.max_vehicles, model=args.model, flags=flags, archive_windows=args.archive_windows)

    print("\nModelo finalizado correctamente:", args.model)
    print("Cuando haya evaluado los tres modelos, ejecute:")
    print(f"python scripts/run_colab_safe.py --config {config_path} --mode {args.mode} --run-name {run_name} --compare-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
