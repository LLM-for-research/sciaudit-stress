"""Метрики §12, которых не хватало: матрица ошибок, калибровка, теги, стресс, композит.

Общая мысль этих проверок: каждая метрика обязана отвечать на свой вопрос и не
подменять соседнюю. Accuracy говорит «сколько», матрица — «куда», калибровка —
«можно ли верить уверенности», стресс-срезы — «что ломается», композит — ничего
научного, только внутренний лидерборд.
"""
import json

import pytest

from sciaudit.evaluator import score as ev


def _pred(instance_id, verdict, confidence=0.9, tags=(), eids=("e01",), abstain=False):
    return {"instance_id": instance_id, "verdict": verdict, "confidence": confidence,
            "predicted_eids": list(eids), "issue_tags": list(tags),
            "abstain": abstain, "rationale_short": "тест",
            "runtime_seconds": 1.0,
            "estimated_cost": {"gpu_seconds": 0.0, "api_cost_usd": 0.0},
            "system_info": {"model": "test", "uses_numeric_checker": False,
                            "uses_lora": False, "uses_vlm": False}}


def _gold(instance_id, verdict, tags=(), eids=("e01",)):
    return {"instance_id": instance_id,
            "gold": {"verdict": verdict, "supporting_eids": list(eids),
                     "issue_tags": list(tags)}}


def _tables(rows):
    """rows: (id, gold_verdict, pred_verdict, confidence, gold_tags, pred_tags)"""
    golds, preds = {}, {}
    for instance_id, gold_verdict, pred_verdict, confidence, gold_tags, pred_tags in rows:
        golds[instance_id] = ev.normalize_gold(_gold(instance_id, gold_verdict,
                                                     gold_tags))
        preds[instance_id] = _pred(instance_id, pred_verdict, confidence, pred_tags)
    return golds, preds, sorted(golds)


# --- §12.1 матрица ошибок -----------------------------------------------------------

def test_the_confusion_matrix_says_where_the_errors_go():
    """Агрегаты говорят сколько, матрица — куда. Это разные болезни."""
    golds, preds, ids = _tables([
        ("a", "warranted", "warranted", 0.9, (), ()),
        ("b", "warranted", "overclaimed", 0.9, (), ()),
        ("c", "overclaimed", "warranted", 0.9, (), ()),
    ])
    matrix = ev.confusion_matrix(golds, preds, ids)
    assert matrix["warranted"]["warranted"] == 1
    assert matrix["warranted"]["overclaimed"] == 1
    assert matrix["overclaimed"]["warranted"] == 1
    assert sum(sum(row.values()) for row in matrix.values()) == 3


# --- §12.4 калибровка ---------------------------------------------------------------

def test_a_perfectly_calibrated_system_has_almost_no_ece():
    rows = [(f"i{n}", "warranted", "warranted" if n < 8 else "overclaimed", 0.8, (), ())
            for n in range(10)]
    golds, preds, ids = _tables(rows)
    result = ev.calibration(golds, preds, ids, bins=10)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)
    assert result["brier"] == pytest.approx(0.8 * 0.04 + 0.2 * 0.64, abs=1e-9)


def test_confident_and_wrong_is_the_worst_case():
    rows = [(f"i{n}", "warranted", "overclaimed", 1.0, (), ()) for n in range(5)]
    golds, preds, ids = _tables(rows)
    result = ev.calibration(golds, preds, ids)
    assert result["brier"] == pytest.approx(1.0)
    assert result["ece"] == pytest.approx(1.0)


def test_calibration_ignores_nothing_and_reports_its_denominator():
    golds, preds, ids = _tables([("a", "warranted", "warranted", 0.5, (), ())])
    assert ev.calibration(golds, preds, ids)["n"] == 1
    assert ev.calibration({}, {}, [])["brier"] is None


# --- §12.3 теги ---------------------------------------------------------------------

def test_macro_and_micro_diverge_when_a_rare_tag_is_missed():
    """Ровно та информация, ради которой мануал требует обе метрики."""
    rows = [(f"i{n}", "overclaimed", "overclaimed", 0.9,
             ("claim_stronger_than_evidence",), ("claim_stronger_than_evidence",))
            for n in range(9)]
    rows.append(("rare", "contradicted", "contradicted", 0.9,
                 ("numerical_inconsistency",), ()))
    golds, preds, ids = _tables(rows)

    tags = ev.issue_tag_metrics(golds, preds, ids)
    assert tags["micro"]["f1"] > tags["macro_f1"]
    assert tags["per_tag"]["numerical_inconsistency"]["f1"] == 0.0
    assert tags["most_missed"][0]["tag"] == "numerical_inconsistency"


# --- §12.6 стресс -------------------------------------------------------------------

def _stress_corpus():
    rows = [
        ("clean1", "warranted", "warranted", 0.9, (), ()),
        ("clean2", "warranted", "warranted", 0.9, (), ()),
        ("stress1", "overclaimed", "warranted", 0.9, (), ()),
        ("stress2", "overclaimed", "overclaimed", 0.9, (), ()),
    ]
    golds, preds, ids = _tables(rows)
    stress = {
        "clean1": {"is_stress_case": False, "stress_type": None},
        "clean2": {"is_stress_case": False, "stress_type": None},
        "stress1": {"is_stress_case": True, "stress_type": "claim_strengthening"},
        "stress2": {"is_stress_case": True, "stress_type": "scope_expansion"},
    }
    return golds, preds, ids, stress


def test_clean_and_stress_are_reported_apart():
    golds, preds, ids, stress = _stress_corpus()
    report = ev.stress_report(golds, preds, ids, stress)
    assert report["clean"]["instances"] == 2
    assert report["stress"]["instances"] == 2
    assert report["clean"]["accuracy"] == pytest.approx(1.0)
    assert report["stress"]["accuracy"] == pytest.approx(0.5)


def test_degradation_has_the_sign_that_reads_as_worse():
    """Падение accuracy положительно, рост SFWR положителен — обе «хуже» в плюс."""
    golds, preds, ids, stress = _stress_corpus()
    degradation = ev.stress_report(golds, preds, ids, stress)["degradation"]
    assert degradation["accuracy"] == pytest.approx(0.5)
    assert degradation["severe_false_warrant_rate"] == pytest.approx(0.5)


def test_stress_type_slices_are_separate():
    golds, preds, ids, stress = _stress_corpus()
    by_type = ev.stress_report(golds, preds, ids, stress)["by_stress_type"]
    assert set(by_type) == {"claim_strengthening", "scope_expansion"}
    assert by_type["claim_strengthening"]["accuracy"] == pytest.approx(0.0)


def test_abstention_is_reported_by_stress_type():
    """Требование §12.5, которое без приватных метаданных посчитать нельзя."""
    golds, preds, ids, stress = _stress_corpus()
    preds["stress1"]["abstain"] = True
    report = ev.stress_report(golds, preds, ids, stress)
    assert report["abstention_by_stress_type"]["claim_strengthening"]["rate"] == 1.0
    assert report["abstention_by_stress_type"]["clean"]["rate"] == 0.0


def test_without_private_metadata_the_slices_are_absent_not_faked():
    golds, preds, ids, _ = _stress_corpus()
    assert ev.stress_report(golds, preds, ids, {}) is None


def test_stress_metadata_is_read_from_private_annotations(tmp_path):
    path = tmp_path / "internal.jsonl"
    path.write_text(json.dumps({
        "instance_id": "sas_abcd1234",
        "stress": {"is_stress_case": True, "stress_type": "evidence_removal",
                   "seed_instance_id": "sas_zzzz9999"}}) + "\n", encoding="utf-8")
    stress = ev.read_stress([path])
    assert stress["sas_abcd1234"]["stress_type"] == "evidence_removal"
    assert stress["sas_abcd1234"]["seed_instance_id"] == "sas_zzzz9999"


# --- §12.7 композит -----------------------------------------------------------------

def test_the_weights_are_the_ones_from_the_manual():
    assert ev.COMPOSITE_WEIGHTS == {
        "verdict_f1": 0.30, "evidence_f1": 0.20, "issue_f1": 0.15,
        "safety": 0.15, "calibration": 0.10, "cost": 0.10}
    assert sum(ev.COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)


def test_a_perfect_system_scores_one():
    metrics = {
        "verdict": {"macro_f1": 1.0},
        "evidence": {"f1": 1.0},
        "issue_tags": {"macro_f1": 1.0},
        "severe_false_warrant_rate_non_abstained": {"rate": 0.0},
        "calibration": {"ece": 0.0},
        "cost": {"cost_norm": 0.0, "cost_reported": True},
    }
    composite = ev.composite_score(metrics)
    assert composite["score"] == pytest.approx(1.0)
    assert composite["weight_covered"] == pytest.approx(1.0)
    assert composite["internal_only"] is True


def test_a_missing_component_is_not_silently_zero():
    """Ноль в формуле — это утверждение. Непосчитанное слагаемое им не заменяется."""
    metrics = {
        "verdict": {"macro_f1": 1.0},
        "evidence": {"f1": 1.0},
        "issue_tags": {"macro_f1": 1.0},
        "severe_false_warrant_rate_non_abstained": {"rate": 0.0},
        "calibration": {"ece": 0.0},
        "cost": {"cost_norm": None, "cost_reported": False},
    }
    composite = ev.composite_score(metrics)
    assert composite["components"]["cost"] is None
    assert composite["weight_covered"] == pytest.approx(0.9)
    assert composite["score"] == pytest.approx(1.0)   # пересчитан на доступные
    assert any("стоимости не посчитано" in note for note in composite["notes"])


def test_safety_and_calibration_actually_lower_the_score():
    metrics = {
        "verdict": {"macro_f1": 1.0}, "evidence": {"f1": 1.0},
        "issue_tags": {"macro_f1": 1.0},
        "severe_false_warrant_rate_non_abstained": {"rate": 1.0},
        "calibration": {"ece": 1.0},
        "cost": {"cost_norm": 0.0, "cost_reported": True},
    }
    assert ev.composite_score(metrics)["score"] == pytest.approx(0.75)


# --- стоимость ----------------------------------------------------------------------

def test_cost_norm_is_undefined_without_a_budget():
    preds = {"a": _pred("a", "warranted")}
    summary = ev.cost_summary(preds, ["a"], budget_usd=None)
    assert summary["cost_norm"] is None
    assert summary["cost_reported"] is False


def test_cost_norm_is_clamped_to_one():
    preds = {"a": _pred("a", "warranted")}
    preds["a"]["estimated_cost"] = {"gpu_seconds": 0.0, "api_cost_usd": 5.0}
    summary = ev.cost_summary(preds, ["a"], budget_usd=1.0)
    assert summary["cost_norm"] == pytest.approx(1.0)
    assert summary["cost_reported"] is True


# --- сквозной прогон ----------------------------------------------------------------

def test_score_carries_every_section_12_block(tmp_path):
    pred_path = tmp_path / "p.jsonl"
    gold_path = tmp_path / "g.jsonl"
    stress_path = tmp_path / "s.jsonl"

    rows = [("sas_0000000%d" % n, "warranted" if n % 2 else "overclaimed") for n in range(6)]
    pred_path.write_text("\n".join(
        json.dumps(_pred(instance_id, verdict)) for instance_id, verdict in rows) + "\n",
        encoding="utf-8")
    gold_path.write_text("\n".join(
        json.dumps(_gold(instance_id, verdict)) for instance_id, verdict in rows) + "\n",
        encoding="utf-8")
    stress_path.write_text("\n".join(
        json.dumps({"instance_id": instance_id,
                    "stress": {"is_stress_case": index % 3 == 0,
                               "stress_type": "scope_expansion" if index % 3 == 0 else None}})
        for index, (instance_id, _) in enumerate(rows)) + "\n", encoding="utf-8")

    metrics = ev.score(str(pred_path), str(gold_path), stress_paths=[str(stress_path)])
    for key in ("confusion_matrix", "calibration", "stress", "cost",
                "composite_internal"):
        assert metrics[key] is not None, key
    assert metrics["issue_tags"]["macro_f1"] is not None
    assert "Матрица ошибок" in ev.make_markdown(metrics)
    assert "Калибровка" in ev.make_markdown(metrics)
    assert "Стресс-срезы" in ev.make_markdown(metrics)


def test_an_unscoreable_submission_reports_no_section_12_numbers(tmp_path):
    """Неполный сабмишен не скорится вовсе — новые блоки не исключение."""
    pred_path = tmp_path / "p.jsonl"
    gold_path = tmp_path / "g.jsonl"
    pred_path.write_text(json.dumps(_pred("sas_00000001", "warranted")) + "\n",
                         encoding="utf-8")
    gold_path.write_text("\n".join([
        json.dumps(_gold("sas_00000001", "warranted")),
        json.dumps(_gold("sas_00000002", "overclaimed"))]) + "\n", encoding="utf-8")

    metrics = ev.score(str(pred_path), str(gold_path))
    assert metrics["submission"]["scoreable"] is False
    for key in ("confusion_matrix", "calibration", "stress", "composite_internal"):
        assert metrics[key] is None, key


# --- чтение отчёта человеком ---------------------------------------------------------

def test_the_report_has_exactly_one_issue_tag_section():
    """Два раздела с одним названием — читатель поверит первому попавшемуся."""
    rows = [(f"i{n}", "warranted", "warranted", 0.9, ("weak_statistical_support",),
             ("weak_statistical_support",)) for n in range(4)]
    golds, preds, ids = _tables(rows)
    metrics = {
        "counts": {k: 0 for k in ("gold_instances", "predictions_submitted",
                                  "valid_predictions", "missing_predictions",
                                  "extra_predictions", "duplicate_predictions",
                                  "invalid_predictions", "abstained_predictions",
                                  "non_abstained_predictions")},
        "submission": {"scoreable": True, "errors": []},
        "coverage": {"value": 1.0, "answered": 4, "total": 4},
        "verdict": ev.verdict_metrics(golds, preds, ids),
        "evidence": ev.set_metrics(golds, preds, ids, "supporting_eids",
                                   "predicted_eids"),
        "issue_tags": ev.issue_tag_metrics(golds, preds, ids),
        "abstention_by_gold_verdict": {},
        "severe_false_warrant_rate_non_abstained":
            ev.severe_false_warrant_rate(golds, preds, ids),
        "selective_risk": {"curve": [], "aurc": 0.0, "augrc": 0.0,
                           "aurc_within_voluntary_coverage": 0.0,
                           "max_voluntary_coverage": 1.0,
                           "at_fixed_coverage": {}, "coverage_at_target_sfwr": {}},
        "calibration": ev.calibration(golds, preds, ids),
        "confusion_matrix": ev.confusion_matrix(golds, preds, ids),
        "stress": None,
        "cost": ev.cost_summary(preds, ids),
        "systems": ["test-model"],
    }
    metrics["composite_internal"] = ev.composite_score(metrics)
    report = ev.make_markdown(metrics)

    assert report.count("## Issue-теги") == 1
    assert "Micro precision" in report and "Macro-F1" in report
    assert "Система: test-model" in report


def test_a_system_that_never_says_warranted_is_told_it_gets_safety_for_free():
    """Иначе B0 читается как «безопасная система», хотя он просто ничего не решает."""
    rows = [(f"i{n}", "warranted", "insufficient", 0.25, (), ()) for n in range(4)]
    golds, preds, ids = _tables(rows)
    metrics = {
        "verdict": ev.verdict_metrics(golds, preds, ids),
        "evidence": {"f1": 0.0},
        "issue_tags": {"macro_f1": 0.0},
        "severe_false_warrant_rate_non_abstained": {"rate": 0.0},
        "calibration": {"ece": 0.0},
        "cost": {"cost_norm": 0.0, "cost_reported": True},
    }
    composite = ev.composite_score(metrics)
    assert composite["components"]["safety"] == pytest.approx(1.0)
    assert any("ни разу не сказала warranted" in note for note in composite["notes"])


def test_a_system_that_does_say_warranted_gets_no_such_note():
    rows = [("a", "warranted", "warranted", 0.9, (), ()),
            ("b", "overclaimed", "warranted", 0.9, (), ())]
    golds, preds, ids = _tables(rows)
    metrics = {
        "verdict": ev.verdict_metrics(golds, preds, ids),
        "evidence": {"f1": 1.0},
        "issue_tags": {"macro_f1": 1.0},
        "severe_false_warrant_rate_non_abstained":
            ev.severe_false_warrant_rate(golds, preds, ids),
        "calibration": {"ece": 0.0},
        "cost": {"cost_norm": 0.0, "cost_reported": True},
    }
    assert not any("ни разу не сказала warranted" in note
                   for note in ev.composite_score(metrics)["notes"])


def test_the_report_names_the_system_from_the_predictions(tmp_path):
    pred_path = tmp_path / "p.jsonl"
    gold_path = tmp_path / "g.jsonl"
    rows = [("sas_0000000%d" % n, "warranted") for n in range(3)]
    predictions = []
    for instance_id, verdict in rows:
        prediction = _pred(instance_id, verdict)
        prediction["system_info"]["model"] = "gpt-oss:120b-cloud"
        predictions.append(prediction)
    pred_path.write_text("\n".join(json.dumps(p) for p in predictions) + "\n",
                         encoding="utf-8")
    gold_path.write_text("\n".join(
        json.dumps(_gold(i, v)) for i, v in rows) + "\n", encoding="utf-8")

    metrics = ev.score(str(pred_path), str(gold_path))
    assert metrics["systems"] == ["gpt-oss:120b-cloud"]
    assert "Система: gpt-oss:120b-cloud" in ev.make_markdown(metrics)
