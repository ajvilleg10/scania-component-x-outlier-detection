from scania_outliers.vehicle_level import aggregate_vehicle_scores, classify_vehicle_scores, make_window_predictions


def _find(rows, vehicle_id):
    return next(r for r in rows if r["vehicle_id"] == vehicle_id)


def test_vehicle_score_aggregation():
    rows = make_window_predictions(
        vehicle_ids=["A", "A", "B", "B"],
        scores=[0.1, 0.9, 0.2, 0.3],
        y_true=[1, 1, 0, 0],
        predictions=[0, 1, 0, 0],
    )
    agg = aggregate_vehicle_scores(rows)
    assert {r["vehicle_id"] for r in agg} == {"A", "B"}
    assert _find(agg, "A")["max_score"] == 0.9
    assert _find(agg, "A")["y_true"] == 1
    assert _find(agg, "B")["y_true"] == 0


def test_vehicle_classification():
    rows = make_window_predictions(["A", "B"], [0.9, 0.2], y_true=[1, 0])
    agg = aggregate_vehicle_scores(rows)
    pred = classify_vehicle_scores(agg, threshold=0.5)
    assert [r["is_outlier"] for r in pred] == [1, 0]
