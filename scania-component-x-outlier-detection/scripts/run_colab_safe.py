from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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


def build_log_dir(config: dict, run_name: str | None = None) -> Path:
    outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))
    stamp = run_name or datetime.now().strftime("safe_%Y%m%d_%H%M%S")
    return outputs_dir / "logs" / "colab_safe_runner" / stamp


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
    parser.add_argument("--run-name", default=None, help="Nombre del directorio de logs para esta ejecución.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.mode == "debug":
        update_debug_config(config_path, args.max_vehicles)

    config = load_config(config_path)
    logs_dir = build_log_dir(config, args.run_name)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\nPLAN DE EJECUCIÓN SEGURA")
    print("Config:", config_path)
    print("Modo:", args.mode)
    print("Modelo:", args.model)
    print("Preparar solamente:", args.prepare_only)
    print("Logs:", logs_dir)
    print("\nNota: este runner entrena un único modelo por ejecución para reducir caídas en Colab.")

    py = sys.executable

    if args.compare_only:
        run_command([py, "main.py", "--config", str(config_path), "--stage", "compare", "--model", "all", "--mode", args.mode], logs_dir / "99_compare.log")
        return 0

    if args.train_only:
        run_command([py, "main.py", "--config", str(config_path), "--stage", "train", "--model", args.model, "--mode", args.mode], logs_dir / f"07_train_{args.model}.log")
        return 0

    if args.evaluate_only:
        run_command([py, "main.py", "--config", str(config_path), "--stage", "evaluate", "--model", args.model, "--mode", args.mode], logs_dir / f"08_evaluate_{args.model}.log")
        return 0

    run_command([py, "scripts/create_drive_folders.py", "--config", str(config_path)], logs_dir / "01_create_drive_folders.log")

    if not args.skip_download:
        run_command([py, "scripts/download_kaggle_to_drive.py", "--config", str(config_path)], logs_dir / "02_download_kaggle_to_drive.log")

    run_command([py, "scripts/check_raw_files.py", "--config", str(config_path)], logs_dir / "03_check_raw_files.log")
    run_command([py, "main.py", "--config", str(config_path), "--stage", "check-data", "--mode", args.mode], logs_dir / "04_check_data_stage.log")

    if not args.skip_eda:
        run_command([py, "main.py", "--config", str(config_path), "--stage", "eda", "--mode", args.mode], logs_dir / "05_eda.log")

    if not args.skip_preprocess:
        run_command([py, "main.py", "--config", str(config_path), "--stage", "preprocess", "--mode", args.mode], logs_dir / "06_preprocess_windowing.log")

    if args.prepare_only:
        print("\nPreparación terminada. Ventanas Parquet listas para entrenar modelos por separado.")
        print("Siguiente paso ejemplo:")
        print(f"python scripts/run_colab_safe.py --config {config_path} --mode {args.mode} --model {args.model} --skip-download --skip-eda --skip-preprocess")
        return 0

    run_command([py, "main.py", "--config", str(config_path), "--stage", "train", "--model", args.model, "--mode", args.mode], logs_dir / f"07_train_{args.model}.log")
    run_command([py, "main.py", "--config", str(config_path), "--stage", "evaluate", "--model", args.model, "--mode", args.mode], logs_dir / f"08_evaluate_{args.model}.log")

    print("\nModelo finalizado correctamente:", args.model)
    print("Cuando haya evaluado los tres modelos, ejecute:")
    print(f"python scripts/run_colab_safe.py --config {config_path} --mode {args.mode} --compare-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
