"""Тесты бейзлайна B1 (BM25 + структурированный аудит моделью, issue #13).

Модель не вызывается по-настоящему ни в одном тесте: везде подставная функция
или подставная команда. Отдельный блок в конце сторожит точку входа и словарь
issue-тегов — именно там баги проехали мимо зелёного CI.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sciaudit.baselines.b1_bm25_llm import (
    DEFAULT_MODEL_NAME,
    FALLBACK_RATIONALE,
    ISSUE_TAGS,
    MOCK_MODEL_NAME,
    count_fallbacks,
    main,
    run,
)
from sciaudit.schemas.validate_predictions import validate_prediction_file

REPO = Path(__file__).resolve().parents[1]


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
        "schema_version": "track_a_input_v1",
        "paper_id": "P901",
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


def _model_returning(payload):
    def model_fn(prompt):
        return json.dumps(payload)
    return model_fn


# --- основной путь ------------------------------------------------------------

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
    assert saved[0]["system_info"]["model"] == MOCK_MODEL_NAME

    assert validate_prediction_file(output_path, input_path) == []


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
    assert saved[0]["rationale_short"] == FALLBACK_RATIONALE
    assert count_fallbacks(saved) == 1

    assert validate_prediction_file(output_path, input_path) == []


def test_b1_filters_predicted_eids_by_allowed_evidence_ids(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(
        input_path,
        [make_input(instance_id="sas_test_003", allowed_evidence_ids=["e01"])],
    )

    run(
        input_path=input_path,
        output_path=output_path,
        top_k=2,
        model_fn=_model_returning({
            "verdict": "warranted",
            "confidence": 0.8,
            "predicted_eids": ["e01", "e02"],
            "issue_tags": [],
            "abstain": False,
            "rationale_short": "Evidence supports the claim.",
        }),
    )

    saved = read_jsonl(output_path)

    assert saved[0]["predicted_eids"] == ["e01"]
    assert validate_prediction_file(output_path, input_path) == []


def test_b1_requires_model_configuration(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input(instance_id="sas_test_004")])

    with pytest.raises(ValueError, match="requires either --model-command or model_fn"):
        run(input_path=input_path, output_path=output_path)


def test_b1_declares_a_model_id_and_never_the_command(tmp_path):
    """Команда запуска может содержать ключ; в system_info.model её быть не должно."""
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input(instance_id="sas_test_005")])

    run(
        input_path=input_path,
        output_path=output_path,
        top_k=1,
        model_command="curl -H 'Authorization: Bearer sk-live-SECRET123'",
        model_name="safe-model-id",
        model_fn=_model_returning({
            "verdict": "warranted",
            "confidence": 0.8,
            "predicted_eids": ["e01"],
            "issue_tags": [],
            "abstain": False,
            "rationale_short": "Evidence supports the claim.",
        }),
    )

    saved = read_jsonl(output_path)

    assert saved[0]["system_info"]["model"] == "safe-model-id"
    assert "sk-live" not in json.dumps(saved[0])
    assert "curl" not in json.dumps(saved[0])


# --- регрессии: словарь тегов, точка входа, тихий отказ ------------------------

def test_issue_tags_match_the_prediction_schema():
    """Дубль словаря в модуле обязан совпадать со схемой, иначе выход невалиден."""
    schema = json.loads((REPO / "schemas" / "prediction.schema.json").read_text("utf-8"))
    assert ISSUE_TAGS == frozenset(schema["properties"]["issue_tags"]["items"]["enum"])


def test_tags_outside_the_vocabulary_are_dropped(tmp_path):
    """Регрессия: теги от модели не проверялись по enum, и выход не проходил схему."""
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"

    write_jsonl(input_path, [make_input(instance_id="sas_test_006")])

    run(
        input_path=input_path,
        output_path=output_path,
        top_k=2,
        model_fn=_model_returning({
            "verdict": "overclaimed",
            "confidence": 0.9,
            "predicted_eids": ["e01"],
            "issue_tags": ["model_parse_failure", "totally_made_up",
                           "claim_stronger_than_evidence"],
            "abstain": False,
            "rationale_short": "x",
        }),
    )

    saved = read_jsonl(output_path)

    assert saved[0]["issue_tags"] == ["claim_stronger_than_evidence"]
    assert validate_prediction_file(output_path, input_path) == []


def test_out_of_range_confidence_is_clamped(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input(instance_id="sas_test_007")])

    run(input_path=input_path, output_path=output_path, top_k=2,
        model_fn=_model_returning({
            "verdict": "warranted", "confidence": 7.5, "predicted_eids": [],
            "issue_tags": [], "abstain": False, "rationale_short": "x"}))

    assert read_jsonl(output_path)[0]["confidence"] == 1.0
    assert validate_prediction_file(output_path, input_path) == []


def test_cli_entry_point_runs(tmp_path):
    """Регрессия: main() падал с UnboundLocalError, а тесты этого не видели.

    Ни один тест не вызывал точку входа, поэтому CI оставался зелёным над
    нерабочей командой. Здесь она запускается ровно так, как её запускает
    человек и как её будет запускать CI.
    """
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input(instance_id="sas_test_008")])

    fake_model = tmp_path / "fake_model.py"
    fake_model.write_text(
        "import sys, json\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'verdict': 'warranted', 'confidence': 0.6,\n"
        "                  'predicted_eids': ['e01'], 'issue_tags': [],\n"
        "                  'abstain': False, 'rationale_short': 'ok'}))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "sciaudit.baselines.b1_bm25_llm",
         "--input", str(input_path), "--output", str(output_path),
         "--top-k", "2",
         "--model-command", f"{sys.executable} {fake_model}",
         "--model-name", "fake-open-model"],
        capture_output=True, text=True, cwd=REPO,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    saved = read_jsonl(output_path)
    assert saved[0]["verdict"] == "warranted"
    assert saved[0]["system_info"]["model"] == "fake-open-model"
    assert validate_prediction_file(output_path, input_path) == []


def test_cli_help_is_reachable():
    result = subprocess.run(
        [sys.executable, "-m", "sciaudit.baselines.b1_bm25_llm", "--help"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert "--model-name" in result.stdout


def test_total_fallback_run_exits_nonzero(tmp_path):
    """Мёртвая модель не должна выглядеть как очень осторожная система."""
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input(instance_id="sas_test_009"),
                             make_input(instance_id="sas_test_010")])

    code = main([
        "--input", str(input_path), "--output", str(output_path),
        "--model-command", "this-command-does-not-exist-12345",
        "--retries", "0",
    ])

    assert code == 1
    saved = read_jsonl(output_path)
    assert count_fallbacks(saved) == len(saved) == 2
    assert validate_prediction_file(output_path, input_path) == []


def test_partial_fallback_run_still_succeeds(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input(instance_id="sas_test_011"),
                             make_input(instance_id="sas_test_012")])

    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps({"verdict": "warranted", "confidence": 0.5,
                           "predicted_eids": ["e01"], "issue_tags": [],
                           "abstain": False, "rationale_short": "ok"})

    predictions = run(input_path=input_path, output_path=output_path,
                      retries=0, model_fn=flaky)

    assert count_fallbacks(predictions) == 1
    assert validate_prediction_file(output_path, input_path) == []


def test_invalid_top_k_is_rejected(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    write_jsonl(input_path, [make_input(instance_id="sas_test_013")])
    with pytest.raises(ValueError, match="top-k"):
        run(input_path=input_path, output_path=tmp_path / "o.jsonl",
            top_k=0, model_fn=_model_returning({}))


def test_bm25_ranking_is_deterministic(tmp_path):
    from sciaudit.baselines.b1_bm25_llm import bm25_rank
    pack = make_input()["evidence_pack"]
    first = [e["eid"] for e in bm25_rank("Dataset A accuracy", pack, top_k=2)]
    for _ in range(5):
        assert [e["eid"] for e in bm25_rank("Dataset A accuracy", pack, top_k=2)] == first
    assert first[0] == "e01"


def test_default_model_name_is_a_model_id_not_a_command():
    assert " " not in DEFAULT_MODEL_NAME
