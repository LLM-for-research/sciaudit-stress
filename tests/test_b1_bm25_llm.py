import json

import pytest

from sciaudit.baselines.b1_bm25_llm import run
from sciaudit.schemas.validate_predictions import validate_prediction_file


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_input(instance_id="sas_test_001", allowed_evidence_ids=None):
    if allowed_evidence_ids is None:
        allowed_evidence_ids = ["e01", "e02"]

    return {
        "schema_version": "student_input_v1",
        "instance_id": instance_id,
        "claim": {
            "text": "Method X improves accuracy on Dataset A.",
            "claim_type": "baseline_superiority",
            "scope": "Judge only from supplied evidence.",
        },
        "allowed_evidence_ids": allowed_evidence_ids,
        "evidence_pack": [
            {
                "eid": "e01",
                "source_kind": "table",
                "modality": "table_text",
                "text": "Dataset A accuracy: Method X = 91.2, Baseline = 89.7.",
            },
            {
                "eid": "e02",
                "source_kind": "table",
                "modality": "table_text",
                "text": "Dataset B accuracy: Method X = 84.1, Baseline = 85.0.",
            },
        ],
    }


def test_b1_outputs_valid_prediction_with_mocked_model(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input()])

    def mock_model(prompt):
        assert "e01" in prompt
        return json.dumps(
            {
                "verdict": "warranted",
                "confidence": 0.8,
                "predicted_eids": ["e01", "fake_eid"],
                "issue_tags": [],
                "abstain": False,
                "rationale_short": "Evidence e01 supports the claim.",
            }
        )

    predictions = run(
        input_path=input_path,
        output_path=output_path,
        top_k=1,
        model_fn=mock_model,
    )

    saved = read_jsonl(output_path)

    assert predictions == saved
    assert saved[0]["instance_id"] == "sas_test_001"
    assert saved[0]["verdict"] == "warranted"
    assert saved[0]["predicted_eids"] == ["e01"]
    assert saved[0]["runtime_seconds"] >= 0
    assert saved[0]["system_info"]["model"] == "mock_model"

    result = validate_prediction_file(output_path, input_path)
    assert result == []


def test_b1_falls_back_after_invalid_model_json_and_stays_schema_valid(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input(instance_id="sas_test_002")])

    def bad_model(prompt):
        return "this is not json"

    run(
        input_path=input_path,
        output_path=output_path,
        top_k=1,
        retries=1,
        model_fn=bad_model,
    )

    saved = read_jsonl(output_path)

    assert saved[0]["verdict"] == "insufficient"
    assert saved[0]["abstain"] is True
    assert saved[0]["predicted_eids"] == []
    assert saved[0]["issue_tags"] == []
    assert "fallback" in saved[0]["rationale_short"]

    result = validate_prediction_file(output_path, input_path)
    assert result == []


def test_b1_filters_predicted_eids_by_allowed_evidence_ids(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(
        input_path,
        [
            make_input(
                instance_id="sas_test_003",
                allowed_evidence_ids=["e01"],
            )
        ],
    )

    def mock_model(prompt):
        return json.dumps(
            {
                "verdict": "warranted",
                "confidence": 0.8,
                "predicted_eids": ["e01", "e02"],
                "issue_tags": [],
                "abstain": False,
                "rationale_short": "Evidence supports the claim.",
            }
        )

    run(
        input_path=input_path,
        output_path=output_path,
        top_k=2,
        model_fn=mock_model,
    )

    saved = read_jsonl(output_path)

    assert saved[0]["predicted_eids"] == ["e01"]

    result = validate_prediction_file(output_path, input_path)
    assert result == []


def test_b1_requires_model_configuration(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input(instance_id="sas_test_004")])

    with pytest.raises(ValueError, match="requires either --model-command or model_fn"):
        run(input_path=input_path, output_path=output_path)


def test_b1_uses_safe_model_name_instead_of_model_command(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input(instance_id="sas_test_005")])

    def mock_model(prompt):
        return json.dumps(
            {
                "verdict": "warranted",
                "confidence": 0.8,
                "predicted_eids": ["e01"],
                "issue_tags": [],
                "abstain": False,
                "rationale_short": "Evidence supports the claim.",
            }
        )

    run(
        input_path=input_path,
        output_path=output_path,
        top_k=1,
        model_command="curl -H 'Authorization: Bearer sk-live-SECRET123'",
        model_name="safe-model-id",
        model_fn=mock_model,
    )

    saved = read_jsonl(output_path)

    assert saved[0]["system_info"]["model"] == "mock_model"
    assert "sk-live" not in json.dumps(saved[0])
