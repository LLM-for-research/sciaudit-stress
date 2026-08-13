"""Тесты evaluator'а Track A (мануал §12).

Покрывают счётчики, качество вердикта, severe false-warrant rate и selective
risk на учебных данных, а также две краевые ситуации: когда система отказалась
от ответа везде и когда нигде.
"""
import json

from sciaudit.evaluator.score import score


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def make_prediction(instance_id, verdict, confidence=0.8, abstain=False):
    return {
        "instance_id": instance_id,
        "verdict": verdict,
        "confidence": confidence,
        "predicted_eids": [],
        "issue_tags": [],
        "abstain": abstain,
    }


def test_track_a_evaluator_on_toy_data():
    metrics = score("examples/toy_predictions.jsonl", "examples/toy_gold.jsonl")

    assert metrics["counts"]["gold_instances"] == 4
    assert metrics["counts"]["missing_predictions"] == 1
    assert metrics["counts"]["non_abstained_predictions"] == 3

    assert "accuracy" in metrics["verdict"]
    assert "macro_f1" in metrics["verdict"]

    assert "f1" in metrics["evidence"]
    assert "f1" in metrics["issue_tags"]

    assert metrics["severe_false_warrant_rate_non_abstained"]["count"] == 1
    assert metrics["severe_false_warrant_rate_non_abstained"]["denominator"] == 2

    assert "aurc" in metrics["selective_risk"]
    assert "thresholds" in metrics["selective_risk"]
    assert "coverage_at_target_sfwr" in metrics["selective_risk"]


def test_all_predictions_abstain(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"

    write_jsonl(
        gold_path,
        [
            {
                "instance_id": "sas_001",
                "gold": {
                    "verdict": "warranted",
                    "supporting_eids": ["e01"],
                    "issue_tags": [],
                },
            },
            {
                "instance_id": "sas_002",
                "gold": {
                    "verdict": "overclaimed",
                    "supporting_eids": ["e02"],
                    "issue_tags": ["claim_stronger_than_evidence"],
                },
            },
        ],
    )

    write_jsonl(
        pred_path,
        [
            make_prediction("sas_001", "warranted", confidence=0.9, abstain=True),
            make_prediction("sas_002", "insufficient", confidence=0.8, abstain=True),
        ],
    )

    metrics = score(pred_path, gold_path)

    assert metrics["counts"]["abstained_predictions"] == 2
    assert metrics["counts"]["non_abstained_predictions"] == 0

    assert metrics["verdict"]["accuracy_denominator"] == 0
    assert metrics["verdict"]["accuracy"] == 0.0

    assert metrics["severe_false_warrant_rate_non_abstained"]["count"] == 0
    assert metrics["severe_false_warrant_rate_non_abstained"]["denominator"] == 0
    assert metrics["severe_false_warrant_rate_non_abstained"]["rate"] == 0.0

    for point in metrics["selective_risk"]["thresholds"]:
        assert point["coverage"] == 0.0
        assert point["risk"] == 0.0
        assert point["selected_count"] == 0


def test_no_predictions_abstain(tmp_path):
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"

    write_jsonl(
        gold_path,
        [
            {
                "instance_id": "sas_001",
                "gold": {
                    "verdict": "warranted",
                    "supporting_eids": ["e01"],
                    "issue_tags": [],
                },
            },
            {
                "instance_id": "sas_002",
                "gold": {
                    "verdict": "overclaimed",
                    "supporting_eids": ["e02"],
                    "issue_tags": ["claim_stronger_than_evidence"],
                },
            },
        ],
    )

    write_jsonl(
        pred_path,
        [
            make_prediction("sas_001", "warranted", confidence=0.9, abstain=False),
            make_prediction("sas_002", "warranted", confidence=0.8, abstain=False),
        ],
    )

    metrics = score(pred_path, gold_path)

    assert metrics["counts"]["abstained_predictions"] == 0
    assert metrics["counts"]["non_abstained_predictions"] == 2

    assert metrics["verdict"]["accuracy_denominator"] == 2
    assert metrics["verdict"]["accuracy"] == 0.5

    assert metrics["severe_false_warrant_rate_non_abstained"]["count"] == 1
    assert metrics["severe_false_warrant_rate_non_abstained"]["denominator"] == 1
    assert metrics["severe_false_warrant_rate_non_abstained"]["rate"] == 1.0

    first_point = metrics["selective_risk"]["thresholds"][0]
    assert first_point["coverage"] == 1.0
    assert first_point["selected_count"] == 2


# --- представление evaluator'а о валидности = prediction.schema.json ----------

def test_prediction_without_abstain_is_scored_not_dropped(tmp_path):
    """`abstain` в схеме необязателен, значит его отсутствие не делает прогноз невалидным.

    Регрессия: evaluator требовал это поле и молча выкидывал схема-валидные
    прогнозы из знаменателя вместо того, чтобы их оценить.
    """
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "pred.jsonl"

    write_jsonl(gold_path, [
        {"instance_id": "sas_001",
         "gold": {"verdict": "warranted", "supporting_eids": ["e01"], "issue_tags": []}},
        {"instance_id": "sas_002",
         "gold": {"verdict": "overclaimed", "supporting_eids": ["e02"], "issue_tags": []}},
    ])
    # минимальная законная форма: только поля, помеченные required в схеме
    write_jsonl(pred_path, [
        {"instance_id": "sas_001", "verdict": "warranted", "confidence": 0.9,
         "predicted_eids": ["e01"], "issue_tags": []},
        {"instance_id": "sas_002", "verdict": "overclaimed", "confidence": 0.7,
         "predicted_eids": ["e02"], "issue_tags": []},
    ])

    metrics = score(pred_path, gold_path)

    assert metrics["counts"]["invalid_predictions"] == 0
    assert metrics["counts"]["valid_predictions"] == 2
    assert metrics["verdict"]["accuracy_denominator"] == 2
    assert metrics["verdict"]["accuracy"] == 1.0


def test_non_boolean_abstain_is_still_rejected():
    from sciaudit.evaluator.score import validate_prediction

    pred = {"instance_id": "sas_001", "verdict": "warranted", "confidence": 0.5,
            "predicted_eids": [], "issue_tags": [], "abstain": "yes"}
    assert any("abstain" in e for e in validate_prediction(pred))


def test_repo_example_predictions_are_all_valid_to_the_evaluator():
    """В examples/sample_predictions.jsonl минимальная форма лежит намеренно."""
    from pathlib import Path
    from sciaudit.evaluator.score import read_jsonl, validate_prediction

    repo = Path(__file__).resolve().parents[1]
    rows = read_jsonl(repo / "examples" / "sample_predictions.jsonl")
    problems = {r["instance_id"]: validate_prediction(r) for r in rows}
    assert not any(problems.values()), problems
