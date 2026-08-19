import pytest

from scania_outliers.labels import DEFAULT_LABEL_CANDIDATES, to_binary_reference_label


def test_class_label_is_first_candidate():
    assert DEFAULT_LABEL_CANDIDATES[0] == "class_label"


def test_default_candidates_include_outlier_terms():
    assert "is_outlier" in DEFAULT_LABEL_CANDIDATES
    assert "in_study_repair" in DEFAULT_LABEL_CANDIDATES


def test_official_temporal_classes_are_collapsed_to_binary_reference():
    assert to_binary_reference_label(0) == 0
    assert [to_binary_reference_label(v) for v in [1, 2, 3, 4]] == [1, 1, 1, 1]
    assert to_binary_reference_label(None) is None
