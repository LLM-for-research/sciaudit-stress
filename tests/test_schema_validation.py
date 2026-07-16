"""Schema + validator tests for Track A (data/schema task, Sprint 1).

Covers the acceptance criteria:
- sample inputs (>=5) validate against track_a_input.schema.json;
- sample predictions (>=5) validate against prediction.schema.json;
- the input validator rejects private fields (gold, verdict, severity,
  stress_type, private_rationale, provenance_map, split, ...);
- the verdict enum is fixed to {warranted, overclaimed, contradicted, insufficient};
- schema enums stay in sync with configs/allowed_labels.yaml;
- the synthetic internal-annotation examples validate against the draft schema;
- both validators run as modules (CLI smoke).
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from sciaudit.schemas import find_forbidden_keys, load_schema, read_jsonl
from sciaudit.schemas.validate_inputs import validate_input_file
from sciaudit.schemas.validate_predictions import validate_prediction_file

REPO = Path(__file__).resolve().parents[1]
SAMPLE_INPUTS = REPO / "examples" / "sample_inputs.jsonl"
SAMPLE_PREDICTIONS = REPO / "examples" / "sample_predictions.jsonl"
SYNTHETIC_ANNOTATIONS = REPO / "examples" / "sample_internal_annotation.synthetic.jsonl"

VERDICTS = ["warranted", "overclaimed", "contradicted", "insufficient"]

PRIVATE_FIELD_SAMPLES = {
    "gold": {"verdict": "overclaimed"},
    "verdict": "overclaimed",
    "severity": "severe",
    "stress_type": "evidence_removal",
    "private_rationale": "leaked",
    "provenance_map": {"P001": "https://example.com/paper"},
    "split": "gold_hidden",
}


def _first_input():
    return read_jsonl(SAMPLE_INPUTS)[0][1]


def _write_jsonl(path, objs):
    path.write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n",
        encoding="utf-8",
    )


# --- example files are valid -------------------------------------------------

def test_sample_inputs_have_at_least_five_instances():
    assert len(read_jsonl(SAMPLE_INPUTS)) >= 5


def test_sample_predictions_have_at_least_five_instances():
    assert len(read_jsonl(SAMPLE_PREDICTIONS)) >= 5


def test_sample_inputs_pass_validator():
    assert validate_input_file(str(SAMPLE_INPUTS)) == []


def test_sample_predictions_pass_validator_with_cross_check():
    problems = validate_prediction_file(str(SAMPLE_PREDICTIONS), str(SAMPLE_INPUTS))
    assert problems == []


def test_synthetic_internal_annotations_match_draft_schema():
    schema = load_schema("internal_annotation.schema.json")
    rows = read_jsonl(SYNTHETIC_ANNOTATIONS)
    assert len(rows) >= 1
    for _, obj in rows:
        jsonschema.validate(obj, schema)


# --- validator rejects private fields ----------------------------------------

@pytest.mark.parametrize("key,value", sorted(PRIVATE_FIELD_SAMPLES.items()))
def test_input_validator_rejects_private_field(tmp_path, key, value):
    obj = copy.deepcopy(_first_input())
    obj[key] = value
    bad = tmp_path / "bad_inputs.jsonl"
    _write_jsonl(bad, [obj])

    problems = validate_input_file(str(bad))
    assert problems, f"private field '{key}' was not rejected"
    assert any("private field" in p or "schema" in p for p in problems)


def test_input_validator_rejects_nested_private_field(tmp_path):
    obj = copy.deepcopy(_first_input())
    obj["evidence_pack"][0]["source_ref"] = "table_2"
    bad = tmp_path / "bad_inputs.jsonl"
    _write_jsonl(bad, [obj])
    assert validate_input_file(str(bad))


def test_forbidden_key_scan_reports_location():
    obj = copy.deepcopy(_first_input())
    obj["gold"] = {"verdict": "warranted"}
    hits = find_forbidden_keys(obj)
    assert any(h.startswith("$.gold") for h in hits)


# --- structural input checks --------------------------------------------------

def test_input_validator_rejects_missing_paper_id(tmp_path):
    obj = copy.deepcopy(_first_input())
    del obj["paper_id"]
    bad = tmp_path / "bad_inputs.jsonl"
    _write_jsonl(bad, [obj])
    assert validate_input_file(str(bad))


def test_input_validator_rejects_unknown_allowed_eid(tmp_path):
    obj = copy.deepcopy(_first_input())
    obj["allowed_evidence_ids"] = obj["allowed_evidence_ids"] + ["e99"]
    bad = tmp_path / "bad_inputs.jsonl"
    _write_jsonl(bad, [obj])
    problems = validate_input_file(str(bad))
    assert any("unknown eid" in p for p in problems)


def test_input_validator_rejects_duplicate_instance_id(tmp_path):
    obj = copy.deepcopy(_first_input())
    bad = tmp_path / "bad_inputs.jsonl"
    _write_jsonl(bad, [obj, obj])
    problems = validate_input_file(str(bad))
    assert any("duplicate instance_id" in p for p in problems)


# --- prediction checks ---------------------------------------------------------

def test_prediction_schema_fixes_verdict_enum():
    schema = load_schema("prediction.schema.json")
    assert schema["properties"]["verdict"]["enum"] == VERDICTS


def test_prediction_validator_rejects_invalid_verdict(tmp_path):
    pred = read_jsonl(SAMPLE_PREDICTIONS)[0][1]
    pred = copy.deepcopy(pred)
    pred["verdict"] = "supported"
    bad = tmp_path / "bad_preds.jsonl"
    _write_jsonl(bad, [pred])
    problems = validate_prediction_file(str(bad))
    assert any("verdict" in p for p in problems)


def test_prediction_validator_rejects_undocumented_issue_tag(tmp_path):
    pred = copy.deepcopy(read_jsonl(SAMPLE_PREDICTIONS)[0][1])
    pred["issue_tags"] = ["made_up_tag"]
    bad = tmp_path / "bad_preds.jsonl"
    _write_jsonl(bad, [pred])
    assert validate_prediction_file(str(bad))


def test_prediction_validator_rejects_eid_outside_allowed(tmp_path):
    pred = copy.deepcopy(read_jsonl(SAMPLE_PREDICTIONS)[0][1])
    pred["predicted_eids"] = ["e99"]
    bad = tmp_path / "bad_preds.jsonl"
    preds = [pred] + [p for _, p in read_jsonl(SAMPLE_PREDICTIONS)[1:]]
    _write_jsonl(bad, preds)
    problems = validate_prediction_file(str(bad), str(SAMPLE_INPUTS))
    assert any("allowed_evidence_ids" in p for p in problems)


def test_minimal_prediction_with_only_required_fields_is_valid(tmp_path):
    pred = {
        "instance_id": "sas_8f3kq2m9",
        "verdict": "insufficient",
        "confidence": 0.5,
        "predicted_eids": [],
        "issue_tags": [],
    }
    good = tmp_path / "min_pred.jsonl"
    _write_jsonl(good, [pred])
    assert validate_prediction_file(str(good)) == []


# --- enums stay in sync with configs/allowed_labels.yaml -----------------------

def test_schema_enums_match_allowed_labels_yaml():
    yaml = pytest.importorskip("yaml")
    labels = yaml.safe_load((REPO / "configs" / "allowed_labels.yaml").read_text(encoding="utf-8"))

    pred_schema = load_schema("prediction.schema.json")
    input_schema = load_schema("track_a_input.schema.json")
    internal_schema = load_schema("internal_annotation.schema.json")

    assert pred_schema["properties"]["verdict"]["enum"] == labels["verdicts"]
    assert pred_schema["properties"]["issue_tags"]["items"]["enum"] == labels["issue_tags"]
    assert (
        input_schema["properties"]["claim"]["properties"]["claim_type"]["enum"]
        == labels["claim_types"]
    )
    gold = internal_schema["properties"]["gold"]["properties"]
    assert gold["verdict"]["enum"] == labels["verdicts"]
    assert gold["issue_tags"]["items"]["enum"] == labels["issue_tags"]


# --- CLI smoke ------------------------------------------------------------------

def test_validate_inputs_cli_ok():
    result = subprocess.run(
        [sys.executable, "-m", "sciaudit.schemas.validate_inputs", str(SAMPLE_INPUTS)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_predictions_cli_ok():
    result = subprocess.run(
        [
            sys.executable, "-m", "sciaudit.schemas.validate_predictions",
            str(SAMPLE_PREDICTIONS), "--input", str(SAMPLE_INPUTS),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_validate_inputs_cli_fails_on_private_fields(tmp_path):
    obj = copy.deepcopy(_first_input())
    obj["gold"] = {"verdict": "warranted", "severity": "severe"}
    bad = tmp_path / "bad_inputs.jsonl"
    _write_jsonl(bad, [obj])
    result = subprocess.run(
        [sys.executable, "-m", "sciaudit.schemas.validate_inputs", str(bad)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 1
