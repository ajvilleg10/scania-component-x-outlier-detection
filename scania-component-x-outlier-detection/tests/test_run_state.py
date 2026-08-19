import json
from pathlib import Path

import pytest

from scania_outliers.run_state import preparation_fingerprint, validate_active_preparation, write_active_preparation


def _config(tmp_path: Path):
    return {
        "execution": {"mode": "debug", "max_vehicles_debug": 25},
        "paths": {"processed_dir": str(tmp_path / "processed")},
        "dataset": {"files": {"train": "train.csv"}},
        "preprocessing": {"max_missing_ratio": 0.8, "fill_strategy": "median", "exclude_columns": []},
        "windowing": {"window_size": 30, "stride": 5, "label_policy": "vehicle_label", "output_format": "parquet"},
    }


def test_preparation_fingerprint_changes_with_debug_size(tmp_path):
    cfg = _config(tmp_path)
    a = preparation_fingerprint(cfg, "debug_025")
    cfg["execution"]["max_vehicles_debug"] = 50
    b = preparation_fingerprint(cfg, "debug_050")
    assert a != b


def test_validate_active_preparation_rejects_other_run(tmp_path):
    cfg = _config(tmp_path)
    write_active_preparation(cfg, "debug_025")
    with pytest.raises(RuntimeError):
        validate_active_preparation(cfg, "debug_050")


def test_validate_active_preparation_accepts_same_run(tmp_path):
    cfg = _config(tmp_path)
    path = write_active_preparation(cfg, "debug_025")
    assert path.exists()
    payload = validate_active_preparation(cfg, "debug_025")
    assert payload["run_name"] == "debug_025"
