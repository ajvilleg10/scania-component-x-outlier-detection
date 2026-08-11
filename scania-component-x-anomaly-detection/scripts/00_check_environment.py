from __future__ import annotations

import argparse
from pathlib import Path

from scania_outliers.config import ensure_directories, load_config


def parse_args():
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

    raw_dir = Path(config["paths"]["raw_dir"])
    alt_raw = Path(config["paths"].get("raw_dir_alternative", ""))
    print("Raw dir exists:", raw_dir.exists(), raw_dir)
    print("Alternative raw dir exists:", alt_raw.exists(), alt_raw)

    print("Configured dataset files:")
    for alias, filename in config["dataset"]["files"].items():
        exists = (raw_dir / filename).exists() or (alt_raw / filename).exists()
        status = "OK" if exists else "MISSING"
        print(f"- {alias}: {filename} [{status}]")


if __name__ == "__main__":
    main()
