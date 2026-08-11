import pandas as pd

from scania_outliers.windowing import TimeWindowBuilder


def test_windowing_builds_expected_shape_with_vehicle_label_policy():
    pdf = pd.DataFrame({
        "vehicle_id": ["A"] * 5 + ["B"] * 5,
        "time_step": [1, 2, 3, 4, 5] * 2,
        "f1": range(10),
        "f2": range(10, 20),
        "y_true": [1] * 5 + [0] * 5,
    })
    builder = TimeWindowBuilder(window_size=3, stride=2, label_policy="vehicle_label")
    data = builder.build_from_pandas(pdf, ["f1", "f2"], label_col="y_true")

    assert data.X.shape == (4, 3, 2)
    assert list(data.y) == [1, 1, 0, 0]
    assert list(data.feature_cols) == ["f1", "f2"]


def test_windowing_max_label_policy_for_true_time_labels():
    pdf = pd.DataFrame({
        "vehicle_id": ["A"] * 5,
        "time_step": [1, 2, 3, 4, 5],
        "f1": range(5),
        "f2": range(10, 15),
        "y_true": [0, 0, 1, 0, 0],
    })
    builder = TimeWindowBuilder(window_size=3, stride=2, label_policy="max_label_in_window")
    data = builder.build_from_pandas(pdf, ["f1", "f2"], label_col="y_true")
    assert list(data.y) == [1, 1]
