"""Тесты B3 — аудита с детерминированной проверкой чисел (мануал §11.5).

Главное, что здесь сторожится: B3 отличается от B2 ровно инструментом. Если
разъедется что-то ещё — селектор evidence, промпт, обработка отказов, — то
сравнение B2 против B3 перестанет измерять вклад проверки чисел и начнёт
измерять разницу реализаций.

Второе по важности — границы полномочий инструмента. Он вправе перебивать
модель, поэтому каждая ситуация, где он этого делать не должен, закрыта
отдельным тестом: сошедшаяся арифметика, отсутствие чисел, мёртвая модель.
"""
import json

import pytest

from sciaudit.baselines import b2_fullpack_llm, b3_checked_llm, model_audit
from sciaudit.baselines.b3_checked_llm import NUMERIC_TAG, NumericTool

CONTRADICTED_PACK = [
    {"eid": "e01", "source_kind": "table_row", "modality": "table_text",
     "text": "Accuracy: Method X = 88.1, Baseline C = 92.1."},
]
CLEAN_PACK = [
    {"eid": "e01", "source_kind": "table_row", "modality": "table_text",
     "text": "Accuracy: Method X = 94.0, Baseline C = 92.1."},
]
PROSE_PACK = [
    {"eid": "e01", "source_kind": "paragraph", "modality": "text",
     "text": "The method generalises to unseen environments."},
]
CONTRADICTED_CLAIM = "Method X outperforms all baselines."


def _instance(pack, claim=CONTRADICTED_CLAIM, instance_id="sas_00000001"):
    return {"instance_id": instance_id,
            "claim": {"text": claim},
            "evidence_pack": pack,
            "allowed_evidence_ids": [u["eid"] for u in pack]}


def _answer(verdict="warranted", **over):
    body = {"verdict": verdict, "confidence": 0.4, "predicted_eids": ["e01"],
            "issue_tags": [], "abstain": False, "rationale_short": "ответ модели"}
    body.update(over)
    return json.dumps(body)


def _model(reply):
    return lambda prompt: reply


# --- промпт ------------------------------------------------------------------------

def test_brief_is_absent_when_there_is_nothing_to_check():
    """Пак без чисел не должен приносить в промпт пустую секцию."""
    tool = NumericTool()
    assert tool.brief("The method generalises well.", PROSE_PACK) is None


def test_brief_carries_the_result_and_the_evidence_ids():
    tool = NumericTool()
    brief = tool.brief(CONTRADICTED_CLAIM, CONTRADICTED_PACK)
    assert "FAILED" in brief
    assert "e01" in brief
    assert "not by a language model" in brief


def test_the_tool_reaches_the_prompt():
    seen = {}

    def model(prompt):
        seen["prompt"] = prompt
        return _answer()

    model_audit.audit_instance(_instance(CONTRADICTED_PACK),
                               b2_fullpack_llm.select_full_pack,
                               model_fn=model, model_name="m", tool=NumericTool())
    assert "Deterministic numeric check" in seen["prompt"]


# --- границы полномочий инструмента --------------------------------------------------

def test_a_numeric_contradiction_overrides_the_model():
    tool = NumericTool()
    prediction, _ = model_audit.audit_instance(
        _instance(CONTRADICTED_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("warranted")), model_name="m", tool=tool)

    assert prediction["verdict"] == "contradicted"
    assert prediction["abstain"] is False
    assert NUMERIC_TAG in prediction["issue_tags"]
    assert prediction["confidence"] == pytest.approx(b3_checked_llm.TOOL_CONFIDENCE)
    assert prediction["rationale_short"].startswith("Numeric check overrode")


def test_matching_arithmetic_does_not_make_a_claim_warranted():
    """Числа умеют опровергать, но не обосновывать.

    Claim может быть переобобщён или вне области применимости — сошедшаяся
    арифметика об этом ничего не говорит, поэтому вердикт модели остаётся.
    """
    prediction, _ = model_audit.audit_instance(
        _instance(CLEAN_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("overclaimed")), model_name="m", tool=NumericTool())
    assert prediction["verdict"] == "overclaimed"
    assert NUMERIC_TAG not in prediction["issue_tags"]


def test_a_pack_without_numbers_leaves_the_verdict_alone():
    prediction, _ = model_audit.audit_instance(
        _instance(PROSE_PACK, claim="The method generalises to every environment."),
        b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("overclaimed")), model_name="m", tool=NumericTool())
    assert prediction["verdict"] == "overclaimed"


def test_the_flag_is_declared_even_when_nothing_is_overridden():
    """system_info должен описывать систему, а не исход конкретного инстанса."""
    prediction, _ = model_audit.audit_instance(
        _instance(PROSE_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer()), model_name="m", tool=NumericTool())
    assert prediction["system_info"]["uses_numeric_checker"] is True


def test_a_dead_model_still_produces_a_plain_safe_fallback():
    """Инструмент не маскирует сбой модели.

    Иначе отказ выглядел бы как уверенный детерминированный ответ, счётчик
    отказов врал бы, а сравнение с B2 перестало бы измерять инструмент.
    """
    def dead(prompt):
        raise model_audit.ModelCallError("модель недоступна")

    prediction, reason = model_audit.audit_instance(
        _instance(CONTRADICTED_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=dead, model_name="m", retries=0, tool=NumericTool())
    assert reason == "модель недоступна"
    assert prediction["abstain"] is True
    assert prediction["verdict"] == "insufficient"
    assert prediction["rationale_short"] == model_audit.FALLBACK_RATIONALE


def test_no_override_mode_advises_without_deciding():
    """Разделяет два вклада инструмента: подсказку в промпте и жёсткое правило."""
    prediction, _ = model_audit.audit_instance(
        _instance(CONTRADICTED_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("warranted")), model_name="m",
        tool=NumericTool(override=False))
    assert prediction["verdict"] == "warranted"
    assert prediction["system_info"]["uses_numeric_checker"] is True


# --- гигиена предсказания ------------------------------------------------------------

def test_tool_evidence_is_merged_not_replaced():
    prediction, _ = model_audit.audit_instance(
        _instance(CONTRADICTED_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("warranted", predicted_eids=[])),
        model_name="m", tool=NumericTool())
    assert prediction["predicted_eids"] == ["e01"]


def test_tool_never_names_evidence_outside_the_pack():
    """Инструмент подчиняется тому же инварианту §5.2, что и модель."""
    instance = _instance(CONTRADICTED_PACK)
    prediction, _ = model_audit.audit_instance(
        instance, lambda claim, pack: [],  # модели показали пустой пак
        model_fn=_model(_answer("warranted")), model_name="m", tool=NumericTool())
    assert prediction["predicted_eids"] == ["e01"]  # только то, что назвала модель
    assert prediction["verdict"] == "warranted"     # проверять было нечего


def test_the_tag_is_not_duplicated():
    prediction, _ = model_audit.audit_instance(
        _instance(CONTRADICTED_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("contradicted", issue_tags=[NUMERIC_TAG])),
        model_name="m", tool=NumericTool())
    assert prediction["issue_tags"].count(NUMERIC_TAG) == 1


def test_a_confident_model_keeps_its_confidence():
    prediction, _ = model_audit.audit_instance(
        _instance(CONTRADICTED_PACK), b2_fullpack_llm.select_full_pack,
        model_fn=_model(_answer("warranted", confidence=0.98)),
        model_name="m", tool=NumericTool())
    assert prediction["confidence"] == pytest.approx(0.98)


# --- контроль относительно B2 --------------------------------------------------------

def test_b3_uses_the_same_evidence_selector_as_b2():
    """Единственная разница обязана быть в инструменте, а не в выборе evidence."""
    assert b3_checked_llm.select_full_pack is b2_fullpack_llm.select_full_pack


def test_without_the_tool_b3_reproduces_b2_exactly(tmp_path):
    inputs = tmp_path / "in.jsonl"
    inputs.write_text(json.dumps(_instance(CLEAN_PACK)) + "\n", encoding="utf-8")

    b2_out, b3_out = tmp_path / "b2.jsonl", tmp_path / "b3.jsonl"
    b2 = b2_fullpack_llm.run(inputs, b2_out, model_fn=_model(_answer()), model_name="m")
    b3 = b3_checked_llm.run(inputs, b3_out, model_fn=_model(_answer()), model_name="m",
                            tool=NumericTool(override=False))

    def comparable(rows):
        return [{k: v for k, v in r.items()
                 if k not in ("runtime_seconds", "system_info")} for r in rows]

    assert comparable(b2) == comparable(b3)


def test_cli_rejects_a_model_command_together_with_the_api():
    assert b3_checked_llm.main(["--input", "i", "--output", "o",
                                "--model-api", "--model-command", "cmd"]) == 2
