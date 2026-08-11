from pathlib import Path

import yaml

from scania_outliers.cli import main


def test_cli_dry_run(tmp_path):
    config = {
        "project": {"name": "test"},
        "execution": {"mode": "debug"},
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "models_dir": str(tmp_path / "models"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
        "modeling": {"models": ["lstm_autoencoder"]},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--config", str(cfg_path), "--dry-run"]) == 0
