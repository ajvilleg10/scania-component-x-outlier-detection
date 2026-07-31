import numpy as np

from scania_anomaly.vehicle_level import aggregate_vehicle_scores, classify_vehicle_scores, make_window_predictions


def test_vehicle_score_aggregation():
    df = make_window_predictions(
        vehicle_ids=["A", "A", "B", "B"],
        scores=[0.1, 0.9, 0.2, 0.3],
        y_true=[1, 1, 0, 0],
        predictions=[0, 1, 0, 0],
    )
    agg = aggregate_vehicle_scores(df)
    assert set(agg["vehicle_id"]) == {"A", "B"}
    assert agg.loc[agg["vehicle_id"] == "A", "max_score"].iloc[0] == 0.9
    assert agg.loc[agg["vehicle_id"] == "A", "y_true"].iloc[0] == 1
    assert agg.loc[agg["vehicle_id"] == "B", "y_true"].iloc[0] == 0


def test_vehicle_classification():
    df = make_window_predictions(["A", "B"], [0.9, 0.2], y_true=[1, 0])
    agg = aggregate_vehicle_scores(df)
    pred = classify_vehicle_scores(agg, threshold=0.5)
    assert pred["is_outlier"].tolist() == [1, 0]
