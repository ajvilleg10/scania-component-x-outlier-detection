from scania_outliers.windowing import TimeWindowBuilder


def test_windowing_builds_expected_shape_with_vehicle_label_policy():
    records = []
    for i in range(5):
        records.append({"vehicle_id": "A", "time_step": i + 1, "f1": i, "f2": i + 10, "y_true": 1})
    for i in range(5):
        records.append({"vehicle_id": "B", "time_step": i + 1, "f1": i + 5, "f2": i + 15, "y_true": 0})
    builder = TimeWindowBuilder(window_size=3, stride=2, label_policy="vehicle_label")
    data = builder.build_from_records(records, ["f1", "f2"], label_col="y_true")

    assert data.X.shape == (4, 3, 2)
    assert list(data.y) == [1, 1, 0, 0]
    assert list(data.feature_cols) == ["f1", "f2"]


def test_windowing_max_label_policy_for_true_time_labels():
    records = [
        {"vehicle_id": "A", "time_step": i + 1, "f1": i, "f2": i + 10, "y_true": y}
        for i, y in enumerate([0, 0, 1, 0, 0])
    ]
    builder = TimeWindowBuilder(window_size=3, stride=2, label_policy="max_label_in_window")
    data = builder.build_from_records(records, ["f1", "f2"], label_col="y_true")
    assert list(data.y) == [1, 1]
