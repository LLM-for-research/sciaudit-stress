from sciaudit.evaluator.score import score


def test_track_a_evaluator_on_toy_data():
    metrics = score("examples/toy_predictions.jsonl", "examples/toy_gold.jsonl")

    assert metrics["counts"]["gold_instances"] == 4
    assert metrics["counts"]["missing_predictions"] == 1

    assert "accuracy" in metrics["verdict"]
    assert "macro_f1" in metrics["verdict"]

    assert "f1" in metrics["evidence"]
    assert "f1" in metrics["issue_tags"]

    assert metrics["severe_false_warrant_rate"]["count"] == 1
    assert metrics["severe_false_warrant_rate"]["denominator"] == 3
