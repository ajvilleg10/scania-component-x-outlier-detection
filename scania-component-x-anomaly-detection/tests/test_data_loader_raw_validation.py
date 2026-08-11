from pathlib import Path

import pytest

from scania_outliers.data_loader import ScaniaDataLoader


class DummySpark:
    pass


def test_missing_files_reports_required_aliases(tmp_path: Path):
    loader = ScaniaDataLoader(
        spark=DummySpark(),
        raw_dir=tmp_path,
        file_map={"train_operational": "train_operational_readouts.csv"},
    )
    missing = loader.missing_files()
    assert missing == {"train_operational": "train_operational_readouts.csv"}


def test_validate_required_files_passes_when_file_exists(tmp_path: Path):
    (tmp_path / "train_operational_readouts.csv").write_text("vehicle_id,time_step\n1,1\n", encoding="utf-8")
    loader = ScaniaDataLoader(
        spark=DummySpark(),
        raw_dir=tmp_path,
        file_map={"train_operational": "train_operational_readouts.csv"},
    )
    loader.validate_required_files()


def test_validate_required_files_fails_with_helpful_message(tmp_path: Path):
    loader = ScaniaDataLoader(
        spark=DummySpark(),
        raw_dir=tmp_path,
        file_map={"validation_labels": "validation_labels.csv"},
    )
    with pytest.raises(FileNotFoundError) as exc:
        loader.validate_required_files()
    assert "download_kaggle_to_drive.py" in str(exc.value)
    assert "validation_labels.csv" in str(exc.value)
