from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from scania_outliers.config import load_config  # noqa: E402
from scania_outliers.study_summary import build_study_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolida métricas entre debug_025/050/100/200 y full sin usar Pandas.")
    parser.add_argument("--config", default="config/config.full.yaml")
    args = parser.parse_args()
    summary = build_study_summary(load_config(args.config))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
