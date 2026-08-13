"""Проверки выхода бейзлайна B0 (мануал §11.2; тест назван в Listing 5).

Убеждаемся, что бейзлайн «всегда insufficient» запускается как модуль и выдаёт
по одному валидному по схеме предсказанию на каждый вход, и в каждом
verdict == "insufficient". Валидация по схеме идёт, только если установлен
`jsonschema`.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLE_INPUTS = REPO / "examples" / "sample_inputs.jsonl"
PRED_SCHEMA = REPO / "schemas" / "prediction.schema.json"
MODULE = "sciaudit.baselines.b0_always_insufficient"

REQUIRED = {
    "instance_id", "verdict", "confidence", "predicted_eids",
    "issue_tags", "abstain", "runtime_seconds", "estimated_cost", "system_info",
}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_b0(out_path):
    subprocess.run(
        [sys.executable, "-m", MODULE, "--input", str(SAMPLE_INPUTS), "--output", str(out_path)],
        check=True,
        cwd=REPO,
    )


def test_b0_emits_one_insufficient_prediction_per_input(tmp_path):
    out = tmp_path / "b0.jsonl"
    _run_b0(out)

    inputs = _read_jsonl(SAMPLE_INPUTS)
    preds = _read_jsonl(out)

    assert len(preds) == len(inputs), "по одному предсказанию на вход"
    assert [p["instance_id"] for p in preds] == [x["instance_id"] for x in inputs]
    for p in preds:
        assert REQUIRED.issubset(p), f"нет ключей: {REQUIRED - set(p)}"
        assert p["verdict"] == "insufficient", "B0 всегда отвечает insufficient"


def test_b0_predictions_match_schema_if_jsonschema_available(tmp_path):
    try:
        import jsonschema
    except ImportError:
        import pytest
        pytest.skip("jsonschema не установлен")

    schema = json.loads(PRED_SCHEMA.read_text(encoding="utf-8"))
    out = tmp_path / "b0.jsonl"
    _run_b0(out)
    for p in _read_jsonl(out):
        jsonschema.validate(p, schema)
