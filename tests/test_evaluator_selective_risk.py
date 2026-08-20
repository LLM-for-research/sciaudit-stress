"""Тесты двух дыр в скоринге: вырождение AURC и неполный сабмишен.

1. При фиксированной сетке порогов система, поставившая всем ответам
   ``confidence = 1.0``, давала одинаковое покрытие во всех точках, площадь под
   кривой вырождалась, и полностью неверная система получала AURC = 0.
2. Пропущенные предсказания не входили в знаменатель accuracy и F1, поэтому
   присылать только уверенные ответы было выгодно.
"""
import json

import pytest

from sciaudit.evaluator.score import (
    FIXED_COVERAGE_LEVELS,
    aurc,
    augrc,
    main,
    risk_coverage_curve,
    score,
)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def gold_row(instance_id, verdict="overclaimed"):
    return {"instance_id": instance_id,
            "gold": {"verdict": verdict, "supporting_eids": [], "issue_tags": []}}


def pred_row(instance_id, verdict, confidence=1.0, abstain=False):
    return {"instance_id": instance_id, "verdict": verdict, "confidence": confidence,
            "predicted_eids": [], "issue_tags": [], "abstain": abstain}


def _score(tmp_path, golds, preds, name="case"):
    gold_path = tmp_path / f"{name}_gold.jsonl"
    pred_path = tmp_path / f"{name}_pred.jsonl"
    write_jsonl(gold_path, golds)
    write_jsonl(pred_path, preds)
    return score(pred_path, gold_path), pred_path, gold_path


# --- 1. вырождение AURC ---------------------------------------------------------

def test_all_confident_and_all_wrong_is_not_rewarded(tmp_path):
    """Регрессия: на пороговой сетке эта система получала AURC = 0."""
    golds = [gold_row(f"sas_{i:03d}") for i in range(10)]
    preds = [pred_row(f"sas_{i:03d}", "warranted", confidence=1.0) for i in range(10)]

    metrics, _, _ = _score(tmp_path, golds, preds)
    selective = metrics["selective_risk"]

    assert metrics["verdict"]["accuracy"] == 0.0
    assert selective["aurc"] == pytest.approx(1.0)
    assert selective["augrc"] == pytest.approx(0.55)


def test_all_confident_and_all_correct_gets_zero(tmp_path):
    golds = [gold_row(f"sas_{i:03d}") for i in range(10)]
    preds = [pred_row(f"sas_{i:03d}", "overclaimed", confidence=1.0) for i in range(10)]

    selective = _score(tmp_path, golds, preds)[0]["selective_risk"]
    assert selective["aurc"] == pytest.approx(0.0)
    assert selective["augrc"] == pytest.approx(0.0)


def test_useful_confidence_beats_misleading_confidence(tmp_path):
    """Одинаковая accuracy, разное ранжирование — метрика обязана их различать."""
    golds = [gold_row(f"sas_{i:03d}") for i in range(10)]
    correct = [f"sas_{i:03d}" for i in range(5)]

    good = [pred_row(i, "overclaimed", 0.9) if i in correct
            else pred_row(i, "warranted", 0.1) for i in [f"sas_{n:03d}" for n in range(10)]]
    bad = [pred_row(i, "overclaimed", 0.1) if i in correct
           else pred_row(i, "warranted", 0.9) for i in [f"sas_{n:03d}" for n in range(10)]]

    good_m = _score(tmp_path, golds, good, "good")[0]
    bad_m = _score(tmp_path, golds, bad, "bad")[0]

    assert good_m["verdict"]["accuracy"] == bad_m["verdict"]["accuracy"] == 0.5
    assert good_m["selective_risk"]["aurc"] < bad_m["selective_risk"]["aurc"]
    assert good_m["selective_risk"]["augrc"] < bad_m["selective_risk"]["augrc"]


def test_ties_are_resolved_by_expectation_not_by_row_order(tmp_path):
    """При равной уверенности порядок строк не должен двигать метрику."""
    golds = [gold_row(f"sas_{i:03d}") for i in range(6)]
    preds = [pred_row("sas_000", "overclaimed"), pred_row("sas_001", "warranted"),
             pred_row("sas_002", "overclaimed"), pred_row("sas_003", "warranted"),
             pred_row("sas_004", "overclaimed"), pred_row("sas_005", "warranted")]

    straight = _score(tmp_path, golds, preds, "straight")[0]["selective_risk"]
    reversed_ = _score(tmp_path, golds, list(reversed(preds)), "rev")[0]["selective_risk"]

    assert straight["aurc"] == pytest.approx(reversed_["aurc"])
    assert straight["augrc"] == pytest.approx(reversed_["augrc"])


def test_generalized_risk_never_exceeds_selective_risk(tmp_path):
    golds = [gold_row(f"sas_{i:03d}") for i in range(8)]
    preds = [pred_row(f"sas_{i:03d}", "warranted" if i % 3 else "overclaimed",
                      confidence=i / 10) for i in range(8)]

    curve = _score(tmp_path, golds, preds)[0]["selective_risk"]["curve"]
    assert all(p["generalized_risk"] <= p["risk"] + 1e-12 for p in curve)
    assert augrc(curve) <= aurc(curve) + 1e-12


def test_curve_has_a_point_for_every_instance(tmp_path):
    golds = [gold_row(f"sas_{i:03d}") for i in range(7)]
    preds = [pred_row(f"sas_{i:03d}", "overclaimed", confidence=i / 10) for i in range(7)]

    curve = _score(tmp_path, golds, preds)[0]["selective_risk"]["curve"]
    assert [p["k"] for p in curve] == list(range(1, 8))
    assert curve[-1]["coverage"] == pytest.approx(1.0)


def test_empty_curve_does_not_crash():
    assert aurc([]) == 0.0 and augrc([]) == 0.0 and risk_coverage_curve({}, {}) == []


# --- метрики при фиксированном покрытии -------------------------------------------

def test_fixed_coverage_levels_are_reported(tmp_path):
    golds = [gold_row(f"sas_{i:03d}") for i in range(10)]
    preds = [pred_row(f"sas_{i:03d}", "overclaimed" if i < 5 else "warranted",
                      confidence=1.0 - i / 10) for i in range(10)]

    fixed = _score(tmp_path, golds, preds)[0]["selective_risk"]["at_fixed_coverage"]

    assert set(fixed) == {str(level) for level in FIXED_COVERAGE_LEVELS}
    assert fixed["0.5"]["achieved_coverage"] == pytest.approx(0.5)
    assert fixed["0.5"]["risk"] == pytest.approx(0.0)      # первые пять верны
    assert fixed["1.0"]["risk"] == pytest.approx(0.5)      # на полном покрытии половина
    assert fixed["1.0"]["achieved_coverage"] == pytest.approx(1.0)


def test_risk_at_full_coverage_equals_the_plain_error_rate(tmp_path):
    golds = [gold_row(f"sas_{i:03d}") for i in range(4)]
    preds = [pred_row("sas_000", "overclaimed", 0.9), pred_row("sas_001", "warranted", 0.8),
             pred_row("sas_002", "overclaimed", 0.7), pred_row("sas_003", "warranted", 0.6)]

    metrics = _score(tmp_path, golds, preds)[0]
    full = metrics["selective_risk"]["at_fixed_coverage"]["1.0"]
    assert full["risk"] == pytest.approx(1 - metrics["verdict"]["accuracy"])


def test_abstention_costs_coverage_and_shows_in_the_forced_tail(tmp_path):
    golds = [gold_row(f"sas_{i:03d}") for i in range(4)]
    preds = [pred_row("sas_000", "overclaimed", 0.9),
             pred_row("sas_001", "overclaimed", 0.8),
             pred_row("sas_002", "insufficient", 0.1, abstain=True),
             pred_row("sas_003", "insufficient", 0.1, abstain=True)]

    metrics = _score(tmp_path, golds, preds)[0]
    selective = metrics["selective_risk"]

    assert metrics["coverage"]["value"] == pytest.approx(0.5)
    assert selective["max_voluntary_coverage"] == pytest.approx(0.5)
    assert selective["aurc_within_voluntary_coverage"] == pytest.approx(0.0)
    # массовый отказ не даёт бесплатного AURC: хвост считается ошибками
    assert selective["aurc"] > 0.0
    assert selective["at_fixed_coverage"]["1.0"]["risk"] == pytest.approx(0.5)


def test_coverage_at_target_sfwr_ignores_the_forced_tail(tmp_path):
    """За границей добровольного покрытия SFWR обманчиво падает: там одни отказы."""
    golds = [gold_row("sas_000"), gold_row("sas_001"), gold_row("sas_002")]
    preds = [pred_row("sas_000", "warranted", 0.9),          # severe false warrant
             pred_row("sas_001", "insufficient", 0.1, abstain=True),
             pred_row("sas_002", "insufficient", 0.1, abstain=True)]

    selective = _score(tmp_path, golds, preds)[0]["selective_risk"]
    assert selective["coverage_at_target_sfwr"]["0.0"] == 0.0


# --- 2. неполный сабмишен ----------------------------------------------------------

def test_partial_submission_of_only_correct_answers_is_rejected(tmp_path):
    """Ровно тот эксплойт: прислать два верных ответа из пяти и получить accuracy 1.0."""
    golds = [gold_row(f"sas_{i:03d}") for i in range(5)]
    preds = [pred_row("sas_000", "overclaimed"), pred_row("sas_001", "overclaimed")]

    metrics, pred_path, gold_path = _score(tmp_path, golds, preds)

    assert metrics["submission"]["scoreable"] is False
    assert metrics["verdict"] is None
    assert metrics["evidence"] is None
    assert metrics["issue_tags"] is None
    assert metrics["selective_risk"] is None
    assert metrics["severe_false_warrant_rate_non_abstained"] is None
    assert metrics["counts"]["missing_predictions"] == 3
    assert any("abstain=true" in e for e in metrics["submission"]["errors"])

    code = main(["--pred", str(pred_path), "--gold", str(gold_path),
                 "--out", str(tmp_path / "m.json")])
    assert code == 1


def test_explicit_abstention_keeps_the_submission_scoreable(tmp_path):
    """Правильный способ выразить неуверенность — отказ, а не пропуск строки."""
    golds = [gold_row(f"sas_{i:03d}") for i in range(5)]
    preds = [pred_row("sas_000", "overclaimed"), pred_row("sas_001", "overclaimed")]
    preds += [pred_row(f"sas_{i:03d}", "insufficient", 0.1, abstain=True)
              for i in range(2, 5)]

    metrics, pred_path, gold_path = _score(tmp_path, golds, preds)

    assert metrics["submission"]["scoreable"] is True
    assert metrics["verdict"]["accuracy"] == 1.0
    assert metrics["verdict"]["accuracy_denominator"] == 2
    assert metrics["coverage"]["value"] == pytest.approx(0.4)

    code = main(["--pred", str(pred_path), "--gold", str(gold_path),
                 "--out", str(tmp_path / "m.json")])
    assert code == 0


def test_extra_predictions_are_rejected(tmp_path):
    golds = [gold_row("sas_000")]
    preds = [pred_row("sas_000", "overclaimed"), pred_row("sas_999", "overclaimed")]

    metrics = _score(tmp_path, golds, preds)[0]
    assert metrics["submission"]["scoreable"] is False
    assert metrics["counts"]["extra_predictions"] == 1


def test_duplicate_predictions_are_rejected(tmp_path):
    golds = [gold_row("sas_000")]
    preds = [pred_row("sas_000", "overclaimed"), pred_row("sas_000", "warranted")]

    metrics = _score(tmp_path, golds, preds)[0]
    assert metrics["submission"]["scoreable"] is False
    assert metrics["counts"]["duplicate_predictions"] == 1
    assert metrics["duplicate_instance_ids"] == ["sas_000"]


def test_invalid_prediction_blocks_scoring(tmp_path):
    golds = [gold_row("sas_000")]
    preds = [{"instance_id": "sas_000", "verdict": "maybe", "confidence": 0.5,
              "predicted_eids": [], "issue_tags": []}]

    metrics = _score(tmp_path, golds, preds)[0]
    assert metrics["submission"]["scoreable"] is False
    assert metrics["counts"]["invalid_predictions"] == 1


def test_rejected_report_says_why_and_prints_no_quality_numbers(tmp_path):
    from sciaudit.evaluator.score import make_markdown

    golds = [gold_row(f"sas_{i:03d}") for i in range(3)]
    preds = [pred_row("sas_000", "overclaimed")]

    report = make_markdown(_score(tmp_path, golds, preds)[0])
    assert "СКОРИНГ ОТКЛОНЁН" in report
    assert "Accuracy" not in report
    assert "AURC" not in report
