"""Тесты B4 — бейзлайна осторожного отказа (мануал §11.6).

Здесь сторожатся две вещи. Первая — каждое из пяти правил отказа срабатывает
тогда и только тогда, когда должно: бейзлайн, который отказывается слишком
охотно, выглядит осторожным, а на деле просто не работает, и по метрикам это
видно не сразу.

Вторая — B4 расходится с B3 ровно в том, что делает при конфликте инструмента с
моделью. B3 перебивает вердикт, B4 отказывается. Если это различие размоется,
исчезнет и смысл держать два бейзлайна.
"""
import json

import pytest

from sciaudit.baselines import b2_fullpack_llm, b3_checked_llm, b4_selective, model_audit
from sciaudit.baselines.b4_selective import (
    RULE_CHECKER_CONFLICT,
    RULE_EMPTY_EVIDENCE,
    RULE_ENSEMBLE_SPLIT,
    RULE_LOW_CONFIDENCE,
    RULE_UNPARSABLE,
    SelectiveAudit,
)

CONTRADICTED_PACK = [
    {"eid": "e01", "source_kind": "table_row", "modality": "table_text",
     "text": "Accuracy: Method X = 88.1, Baseline C = 92.1."},
]
CLEAN_PACK = [
    {"eid": "e01", "source_kind": "table_row", "modality": "table_text",
     "text": "Accuracy: Method X = 94.0, Baseline C = 92.1."},
]
CLAIM = "Method X outperforms all baselines."


def _instance(pack=CLEAN_PACK, claim=CLAIM, instance_id="sas_00000001"):
    return {"instance_id": instance_id,
            "claim": {"text": claim},
            "evidence_pack": pack,
            "allowed_evidence_ids": [u["eid"] for u in pack]}


def _answer(verdict="warranted", confidence=0.95, eids=("e01",), **over):
    body = {"verdict": verdict, "confidence": confidence,
            "predicted_eids": list(eids), "issue_tags": [], "abstain": False,
            "rationale_short": "ответ модели"}
    body.update(over)
    return json.dumps(body)


def _replies(*answers):
    """Модель, отвечающая по очереди. Последний ответ повторяется."""
    queue = list(answers)

    def model(prompt):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return model


def _audit(instance, model, votes=3, threshold=0.9, **over):
    tool = b3_checked_llm.NumericTool(override=False)
    audit = SelectiveAudit(votes=votes, confidence_threshold=threshold, **over)
    prediction, reason = audit(instance, b2_fullpack_llm.select_full_pack,
                               model_fn=model, model_name="m", retries=0, tool=tool)
    return prediction, reason, audit


# --- когда B4 отвечает --------------------------------------------------------------

def test_unanimous_and_confident_is_answered():
    prediction, _, audit = _audit(_instance(), _replies(_answer()))
    assert prediction["abstain"] is False
    assert prediction["verdict"] == "warranted"
    assert audit.counts["answered"] == 1
    assert audit.counts["abstained"] == 0


def test_confidence_is_agreement_times_stated_confidence():
    """Два независимых сигнала неуверенности перемножаются, а не подменяют друг друга."""
    prediction, _, _ = _audit(
        _instance(),
        _replies(_answer(confidence=0.9), _answer(confidence=0.9),
                 _answer("overclaimed", confidence=0.9)),
        threshold=0.0)
    # 2 голоса из 3 за warranted, заявленная уверенность 0.9
    assert prediction["confidence"] == pytest.approx(2 / 3 * 0.9, abs=1e-4)


def test_with_one_vote_the_confidence_is_the_model_own():
    prediction, _, _ = _audit(_instance(), _replies(_answer(confidence=0.77)),
                              votes=1, threshold=0.0)
    assert prediction["confidence"] == pytest.approx(0.77)


def test_an_insufficient_verdict_may_name_no_evidence():
    """«Доказательств не хватает» как раз и означает, что называть нечего."""
    prediction, _, audit = _audit(
        _instance(), _replies(_answer("insufficient", eids=())))
    assert prediction["abstain"] is False
    assert audit.counts[RULE_EMPTY_EVIDENCE] == 0


# --- пять правил §11.6 --------------------------------------------------------------

def test_low_confidence_abstains():
    prediction, _, audit = _audit(_instance(), _replies(_answer(confidence=0.4)))
    assert prediction["abstain"] is True
    assert audit.counts[RULE_LOW_CONFIDENCE] == 1
    assert RULE_LOW_CONFIDENCE in prediction["rationale_short"]


def test_a_decisive_verdict_without_evidence_abstains():
    prediction, _, audit = _audit(_instance(), _replies(_answer("warranted", eids=())))
    assert prediction["abstain"] is True
    assert audit.counts[RULE_EMPTY_EVIDENCE] == 1


def test_a_split_ensemble_abstains():
    prediction, _, audit = _audit(
        _instance(),
        _replies(_answer("warranted"), _answer("overclaimed"), _answer("insufficient")))
    assert prediction["abstain"] is True
    assert audit.counts[RULE_ENSEMBLE_SPLIT] == 1


def test_a_bare_majority_answers_but_unanimity_can_be_required():
    two_of_three = (_answer(), _answer(), _answer("overclaimed"))
    assert _audit(_instance(), _replies(*two_of_three))[0]["abstain"] is False
    strict = _audit(_instance(), _replies(*two_of_three), require_unanimous=True)
    assert strict[0]["abstain"] is True
    assert strict[2].counts[RULE_ENSEMBLE_SPLIT] == 1


def test_a_checker_conflict_abstains():
    prediction, _, audit = _audit(_instance(CONTRADICTED_PACK), _replies(_answer()))
    assert prediction["abstain"] is True
    assert audit.counts[RULE_CHECKER_CONFLICT] == 1


def test_the_checker_agreeing_with_the_model_is_not_a_conflict():
    prediction, _, audit = _audit(_instance(CONTRADICTED_PACK),
                                  _replies(_answer("contradicted")))
    assert prediction["abstain"] is False
    assert audit.counts[RULE_CHECKER_CONFLICT] == 0


def test_a_dead_model_yields_the_plain_safe_fallback():
    """Сбой не переписывается в осторожность: доля отказов обязана остаться честной."""
    def dead(prompt):
        raise model_audit.ModelCallError("модель недоступна")

    prediction, reason, audit = _audit(_instance(), dead)
    assert prediction["abstain"] is True
    assert prediction["rationale_short"] == model_audit.FALLBACK_RATIONALE
    assert model_audit.count_fallbacks([prediction]) == 1
    assert reason == "модель недоступна"
    assert audit.counts[RULE_UNPARSABLE] == 1


def test_a_partial_failure_is_not_counted_as_a_parsing_fallback():
    """Один живой голос из трёх — это ответ модели, но не основание отвечать."""
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return _answer()
        raise model_audit.ModelCallError("timeout")

    prediction, _, audit = _audit(_instance(), flaky)
    assert model_audit.count_fallbacks([prediction]) == 0
    assert prediction["abstain"] is True
    assert audit.counts[RULE_ENSEMBLE_SPLIT] == 1


# --- отчётность ---------------------------------------------------------------------

def test_the_abstention_records_the_verdict_it_gave_up_on():
    prediction, _, _ = _audit(_instance(), _replies(_answer("overclaimed", confidence=0.4)))
    assert prediction["verdict"] == "insufficient"       # общая конвенция проекта
    assert "overclaimed" in prediction["rationale_short"]  # но решение не потеряно


def test_all_fired_rules_are_reported_not_just_the_first():
    prediction, _, audit = _audit(
        _instance(CONTRADICTED_PACK), _replies(_answer(confidence=0.4, eids=())))
    for rule in (RULE_LOW_CONFIDENCE, RULE_EMPTY_EVIDENCE, RULE_CHECKER_CONFLICT):
        assert rule in prediction["rationale_short"]
        assert audit.counts[rule] == 1


def test_a_rule_can_be_switched_off_for_an_ablation():
    """Иначе вклад отдельного правила из одного прогона не отделить."""
    prediction, _, _ = _audit(_instance(CONTRADICTED_PACK), _replies(_answer()),
                              disabled_rules=(RULE_CHECKER_CONFLICT,))
    assert prediction["abstain"] is False


def test_summary_counts_every_call():
    _, _, audit = _audit(_instance(), _replies(_answer()), votes=3)
    assert audit.counts["votes"] == 3
    assert "вызовов модели — 3" in audit.summary()


# --- отношение к другим бейзлайнам --------------------------------------------------

def test_b4_sees_the_same_evidence_as_b2_and_b3():
    assert b4_selective.select_full_pack is b2_fullpack_llm.select_full_pack


def test_b3_overrides_where_b4_abstains():
    """Содержательная разница между двумя защитимыми позициями.

    B3 считает арифметику надёжнее модели и правит вердикт. B4 считает
    расхождение признаком тяжёлого инстанса и снимает с себя ответ. Оба
    поведения нужны, поэтому разведены по разным бейзлайнам.
    """
    instance = _instance(CONTRADICTED_PACK)
    b3_prediction, _ = model_audit.audit_instance(
        instance, b2_fullpack_llm.select_full_pack,
        model_fn=_replies(_answer()), model_name="m",
        tool=b3_checked_llm.NumericTool())
    b4_prediction, _, _ = _audit(instance, _replies(_answer()))

    assert b3_prediction["verdict"] == "contradicted"
    assert b3_prediction["abstain"] is False
    assert b4_prediction["abstain"] is True


def test_the_run_writes_one_prediction_per_input(tmp_path):
    inputs = tmp_path / "in.jsonl"
    inputs.write_text("\n".join(json.dumps(_instance(instance_id=f"sas_0000000{i}"))
                                for i in (1, 2)) + "\n", encoding="utf-8")
    out = tmp_path / "p.jsonl"
    predictions = b4_selective.run(inputs, out, votes=2, model_fn=_replies(_answer()),
                                   model_name="m")
    assert len(predictions) == 2
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_the_number_of_calls_is_votes_times_instances(tmp_path):
    calls = []
    inputs = tmp_path / "in.jsonl"
    inputs.write_text(json.dumps(_instance()) + "\n", encoding="utf-8")

    def counting(prompt):
        calls.append(prompt)
        return _answer()

    b4_selective.run(inputs, tmp_path / "p.jsonl", votes=3, model_fn=counting,
                     model_name="m")
    assert len(calls) == 3


def test_impossible_settings_are_refused():
    with pytest.raises(ValueError):
        SelectiveAudit(votes=0)
    with pytest.raises(ValueError):
        SelectiveAudit(confidence_threshold=1.5)
    assert b4_selective.main(["--input", "i", "--output", "o", "--votes", "0"]) == 2
