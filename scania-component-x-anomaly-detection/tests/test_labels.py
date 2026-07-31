import pytest

from scania_anomaly.labels import DEFAULT_LABEL_CANDIDATES


def test_class_label_is_first_candidate():
    assert DEFAULT_LABEL_CANDIDATES[0] == "class_label"


def test_default_candidates_include_outlier_terms():
    assert "is_outlier" in DEFAULT_LABEL_CANDIDATES
    assert "in_study_repair" in DEFAULT_LABEL_CANDIDATES
