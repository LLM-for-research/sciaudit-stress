"""Smoke-тест: шаблон системы проходит от входа до конца и выдаёт валидные предсказания.

Держит рельсу честной (мануал §1.3, §8.1). Основные проверки используют только
стандартную библиотеку; валидация по схеме идёт, только если установлен
`jsonschema`.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "system_template" / "run.py"
SAMPLE_INPUTS = REPO / "examples" / "sample_inputs.jsonl"
PRED_SCHEMA = REPO / "schemas" / "prediction.schema.json"

VERDICTS = {"warranted", "overclaimed", "contradicted", "insufficient"}
REQUIRED = {
    "instance_id", "verdict", "confidence", "predicted_eids",
    "issue_tags", "abstain", "runtime_seconds", "estimated_cost", "system_info",
}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run(out_path):
    subprocess.run(
        [sys.executable, str(RUN), "--input", str(SAMPLE_INPUTS), "--output", str(out_path)],
        check=True,
    )


def test_template_runs_and_emits_valid_predictions(tmp_path):
    out = tmp_path / "predictions.jsonl"
    _run(out)

    inputs = _read_jsonl(SAMPLE_INPUTS)
    preds = _read_jsonl(out)

    assert len(preds) == len(inputs), "по одному предсказанию на вход"
    assert [p["instance_id"] for p in preds] == [x["instance_id"] for x in inputs], \
        "instance_id сохранены и идут в том же порядке"

    for p in preds:
        assert REQUIRED.issubset(p), f"нет ключей: {REQUIRED - set(p)}"
        assert p["verdict"] in VERDICTS
        assert 0.0 <= p["confidence"] <= 1.0


def test_predictions_match_schema_if_jsonschema_available(tmp_path):
    try:
        import jsonschema
    except ImportError:
        import pytest
        pytest.skip("jsonschema не установлен")

    schema = json.loads(PRED_SCHEMA.read_text(encoding="utf-8"))
    out = tmp_path / "predictions.jsonl"
    _run(out)
    for p in _read_jsonl(out):
        jsonschema.validate(p, schema)
