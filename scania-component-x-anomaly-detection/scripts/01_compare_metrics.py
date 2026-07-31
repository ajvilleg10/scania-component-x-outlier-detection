from pathlib import Path

import pandas as pd

from scania_anomaly.config import load_config


def main() -> None:
    config = load_config()
    metrics_dir = Path(config["paths"]["metrics_dir"])
    files = sorted(metrics_dir.glob("*_test_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No metric files found in {metrics_dir}")

    rows = [pd.read_json(file, typ="series").to_dict() for file in files]
    comparison = pd.DataFrame(rows).sort_values("f1_score", ascending=False)

    output_path = Path(config["paths"]["tables_dir"]) / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(comparison)
    print(f"Saved comparison table to {output_path}")


if __name__ == "__main__":
    main()
