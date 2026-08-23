"""Orchestrates the learning-curve study (Hipótesis específica 1).

Design (see docs/learning_curve.md for the full rationale):
  - For each training size N in --sizes, and each seed in 1..--n-seeds, sample
    N *normal* training vehicles at random (seed-controlled) and train on them.
  - Validation and test are NEVER subsampled here — they always use the full
    official splits, exactly like the `full` run. Only training volume varies,
    so a change in the metric can only be attributed to training volume.
  - Each (N, seed) combination is its own `--run-name` (learning_curve_n{N}_s{seed}),
    reusing the same prepare -> train -> evaluate flow as any other run. The
    Parquet window cache is shared, so a fresh --prepare-only runs before every
    combination, same as switching between debug_025 and debug_050 today.
  - The `full` run (already required as the main experimental result) supplies
    the anchor point at N = all normal training vehicles, with no repeats
    needed there — build_learning_curve_summary.py picks it up automatically.

This trains ONE reference architecture (default: lstm_autoencoder) across the
grid, not all three, to keep this an addition rather than a multiplier on top
of the required debug_025 + full runs. Override with --model if you want the
full grid for another architecture too (increases Colab time roughly linearly
in the number of (size, seed) combinations).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from scania_outliers.config import load_config  # noqa: E402

# Reuse run_command / main_command / save_runner_event from run_colab_safe.py so
# both scripts log and behave identically instead of drifting apart over time.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_colab_safe import raw_files_complete, run_command, save_runner_event  # noqa: E402


def learning_curve_run_name(n_vehicles: int, seed: int) -> str:
    return f"learning_curve_n{n_vehicles:03d}_s{seed}"


def lc_main_command(py: str, config_path: Path, run_name: str, n_vehicles: int, seed: int, stage: str, model: str | None = None) -> list[str]:
    command = [
        py,
        "main.py",
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--mode",
        "learning_curve",
        "--run-id",
        run_name,
        "--learning-curve-n-vehicles",
        str(n_vehicles),
        "--learning-curve-seed",
        str(seed),
    ]
    if model is not None:
        command += ["--model", model]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.full.yaml", help="Debe apuntar a un config con validation/test completos (config.full.yaml).")
    parser.add_argument("--sizes", default="25,50,100,200", help="Lista de N (vehículos de train) separada por comas.")
    parser.add_argument("--n-seeds", type=int, default=3, help="Repeticiones por tamaño, con semillas 1..n_seeds.")
    parser.add_argument("--model", default="lstm_autoencoder", choices=["lstm_autoencoder", "cnn_lstm_autoencoder", "transformer_encoder"])
    parser.add_argument("--force-download", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    experiments_dir = Path(config.get("paths", {}).get("experiments_dir", Path(config["paths"]["drive_root"]) / "experiments"))
    py = sys.executable

    # Raw files only need to exist once; reuse across every combination.
    if args.force_download or not raw_files_complete(config):
        download_cmd = [py, "scripts/download_kaggle_to_drive.py", "--config", str(config_path)]
        if args.force_download:
            download_cmd.append("--force")
        run_command(download_cmd, experiments_dir / "learning_curve_logs" / "00_download_kaggle_to_drive.log")
    run_command([py, "scripts/create_drive_folders.py", "--config", str(config_path)], experiments_dir / "learning_curve_logs" / "01_create_drive_folders.log")
    run_command([py, "scripts/check_raw_files.py", "--config", str(config_path)], experiments_dir / "learning_curve_logs" / "02_check_raw_files.log")

    total = len(sizes) * args.n_seeds
    done = 0
    for n_vehicles in sizes:
        for seed in range(1, args.n_seeds + 1):
            done += 1
            run_name = learning_curve_run_name(n_vehicles, seed)
            run_dir = experiments_dir / "runs" / run_name
            logs_dir = run_dir / "logs"
            print(f"\n[{done}/{total}] N={n_vehicles} seed={seed} run={run_name}")

            save_runner_event(run_dir, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": "learning_curve",
                "n_vehicles": n_vehicles,
                "seed": seed,
                "model": args.model,
            })

            run_command(lc_main_command(py, config_path, run_name, n_vehicles, seed, "check-data"), logs_dir / "04_check_data.log")
            run_command(lc_main_command(py, config_path, run_name, n_vehicles, seed, "eda"), logs_dir / "05_eda.log")
            run_command(lc_main_command(py, config_path, run_name, n_vehicles, seed, "preprocess"), logs_dir / "06_preprocess_windowing.log")
            run_command(lc_main_command(py, config_path, run_name, n_vehicles, seed, "train", args.model), logs_dir / f"10_train_{args.model}.log")
            run_command(lc_main_command(py, config_path, run_name, n_vehicles, seed, "evaluate", args.model), logs_dir / f"20_evaluate_{args.model}.log")
            run_command(lc_main_command(py, config_path, run_name, n_vehicles, seed, "report"), logs_dir / "99_report.log")

    print(f"\nCompletadas {total} combinaciones (tamaños={sizes}, semillas=1..{args.n_seeds}, modelo={args.model}).")
    print("Ejecute ahora: python scripts/build_learning_curve_summary.py --config", str(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
