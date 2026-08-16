from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare JSON metric files without Pandas")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    files = sorted(metrics_dir.glob("*_metrics.json"))
    rows = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r.get("f1_score") is None, -(r.get("f1_score") or 0)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
