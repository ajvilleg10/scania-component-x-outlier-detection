from pathlib import Path

import pytest
import yaml

from scania_outliers.cli import main


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"name": "test"},
        "execution": {"mode": "debug", "max_vehicles_debug": 25},
        "paths": {
            "drive_root": str(tmp_path),
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "experiments_dir": str(tmp_path / "experiments"),
        },
        "modeling": {"models": ["lstm_autoencoder"]},
    }


def test_cli_dry_run(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    assert main(["--config", str(cfg_path), "--dry-run", "--run-id", "test_run"]) == 0
    assert (tmp_path / "experiments" / "runs" / "test_run" / "config_used.yaml").exists()


def test_cli_blocks_all_models_for_heavy_stage_by_default(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--config", str(cfg_path), "--stage", "train", "--model", "all", "--dry-run"])
