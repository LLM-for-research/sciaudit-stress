"""Тесты общего слоя вызова модели (issue #14).

Слой один на все бейзлайны на модели, поэтому инварианты проверяются здесь
один раз, а не копируются в тесты каждого бейзлайна.
"""
import json
from pathlib import Path

import pytest

from sciaudit.baselines import model_audit as ma

REPO = Path(__file__).resolve().parents[1]


def _pack(*texts):
    return [{"eid": f"e{i:02d}", "source_kind": "table_row", "modality": "table_text",
             "text": t} for i, t in enumerate(texts, start=1)]


def _instance(**over):
    base = {
        "instance_id": "sas_shared01",
        "claim": {"text": "Method X improves accuracy.", "claim_type": "ablation",
                  "scope": "Judge only from the supplied evidence pack."},
        "evidence_pack": _pack("Table 1: Method X = 91.2, Baseline = 89.7.",
                               "Table 2: Method X = 84.1, Baseline = 85.0."),
        "allowed_evidence_ids": ["e01", "e02"],
    }
    base.update(over)
    return base


# --- словарь тегов -------------------------------------------------------------

def test_issue_tags_match_the_prediction_schema():
    schema = json.loads((REPO / "schemas" / "prediction.schema.json").read_text("utf-8"))
    assert ma.ISSUE_TAGS == frozenset(schema["properties"]["issue_tags"]["items"]["enum"])


# --- нормализация --------------------------------------------------------------

def _normalize(raw, allowed=("e01", "e02")):
    return ma.normalize_prediction(raw, "sas_shared01", set(allowed), "m", 0.0)


def test_unknown_verdict_becomes_insufficient():
    assert _normalize({"verdict": "probably_fine"})["verdict"] == "insufficient"


def test_confidence_is_clamped_and_booleans_are_rejected():
    assert _normalize({"confidence": 7.5})["confidence"] == 1.0
    assert _normalize({"confidence": -3})["confidence"] == 0.0
    assert _normalize({"confidence": True})["confidence"] == 0.0


def test_eids_outside_allowed_are_dropped_and_order_is_kept():
    out = _normalize({"predicted_eids": ["e02", "nope", "e01", "e02"]})
    assert out["predicted_eids"] == ["e02", "e01"]


def test_tags_outside_the_vocabulary_are_dropped():
    out = _normalize({"issue_tags": ["model_parse_failure", "numerical_inconsistency",
                                     "numerical_inconsistency"]})
    assert out["issue_tags"] == ["numerical_inconsistency"]


def test_non_list_fields_degrade_to_empty():
    out = _normalize({"predicted_eids": "e01", "issue_tags": "numerical_inconsistency"})
    assert out["predicted_eids"] == [] and out["issue_tags"] == []


def test_non_boolean_abstain_becomes_false():
    assert _normalize({"abstain": "yes"})["abstain"] is False


# --- промпт --------------------------------------------------------------------

def test_prompt_lists_only_the_selected_evidence():
    prompt = ma.build_prompt("Claim text.", _pack("first unit", "second unit")[:1])
    assert "e01" in prompt and "e02" not in prompt


def test_prompt_names_the_allowed_tag_vocabulary():
    prompt = ma.build_prompt("Claim text.", _pack("unit"))
    for tag in ma.ISSUE_TAGS:
        assert tag in prompt


# --- вызов модели и отказы ------------------------------------------------------

def test_missing_model_command_is_an_error():
    with pytest.raises(ma.ModelCallError, match="not configured"):
        ma.call_model("prompt", model_command=None)


def test_nonzero_exit_is_reported_with_stderr_tail():
    with pytest.raises(ma.ModelCallError, match="exited with"):
        ma.call_model("prompt", model_command="echo boom >&2; exit 3")


def test_empty_output_is_an_error():
    with pytest.raises(ma.ModelCallError, match="empty"):
        ma.parse_model_json("   ")


def test_json_embedded_in_prose_is_recovered():
    raw = ma.parse_model_json('here you go:\n{"verdict": "warranted"}\nhope that helps')
    assert raw["verdict"] == "warranted"


def test_prose_without_json_is_an_error():
    with pytest.raises(ma.ModelCallError, match="no JSON object"):
        ma.parse_model_json("no braces at all")


def test_audit_instance_reports_the_fallback_reason():
    prediction, reason = ma.audit_instance(
        _instance(), lambda claim, pack: pack, model_fn=lambda p: "not json", retries=0)
    assert prediction["verdict"] == "insufficient"
    assert prediction["abstain"] is True
    assert prediction["rationale_short"] == ma.FALLBACK_RATIONALE
    assert "no JSON object" in reason


def test_instance_without_id_is_rejected_loudly():
    broken = _instance()
    del broken["instance_id"]
    with pytest.raises(ValueError, match="instance_id"):
        ma.audit_instance(broken, lambda claim, pack: pack, model_fn=lambda p: "{}")


def test_allowed_eids_fall_back_to_the_pack_when_absent():
    instance = _instance()
    del instance["allowed_evidence_ids"]
    prediction, _ = ma.audit_instance(
        instance, lambda claim, pack: pack,
        model_fn=lambda p: json.dumps({"verdict": "warranted",
                                       "predicted_eids": ["e02", "ghost"]}))
    assert prediction["predicted_eids"] == ["e02"]


def test_model_id_is_declared_never_derived_from_the_command():
    prediction, _ = ma.audit_instance(
        _instance(), lambda claim, pack: pack,
        model_command="curl -H 'Authorization: Bearer sk-live-SECRET'",
        model_name="declared-model",
        model_fn=lambda p: json.dumps({"verdict": "warranted"}))
    assert prediction["system_info"]["model"] == "declared-model"
    assert "sk-live" not in json.dumps(prediction)


def test_count_fallbacks_counts_only_safe_refusals():
    good, _ = ma.audit_instance(_instance(), lambda c, p: p,
                                model_fn=lambda p: json.dumps({"verdict": "warranted"}))
    bad, _ = ma.audit_instance(_instance(), lambda c, p: p,
                               model_fn=lambda p: "nope", retries=0)
    assert ma.count_fallbacks([good, bad]) == 1
