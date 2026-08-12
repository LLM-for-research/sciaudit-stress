"""Readiness-loop CI guard (issue #11).

The full staff-manual §8.1 chain — inputs -> B0 -> validate -> leakage scan
-> evaluator -> metrics — must run in CI on the public warm-up slice, and no
commented-out "enable later" steps may linger for modules that already exist.
The root README documents the same chain for local runs.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"

WARMUP_INPUTS = "data_public/public_warmup/inputs.jsonl"
WARMUP_GOLD = "data_public/public_warmup/gold.jsonl"

CHAIN_FRAGMENTS = (
    "b0_always_insufficient",
    "sciaudit.schemas.validate_predictions",
    "sciaudit.evaluator.score",
    "sciaudit.leakage.forbidden_key_scan",
)


def _ci_text() -> str:
    return CI.read_text(encoding="utf-8")


def test_ci_runs_b0_on_public_warmup():
    text = _ci_text()
    assert "b0_always_insufficient" in text
    assert WARMUP_INPUTS in text


def test_ci_validates_warmup_predictions():
    text = _ci_text()
    assert "validate_predictions" in text
    assert WARMUP_INPUTS in text


def test_ci_runs_evaluator_with_warmup_gold():
    text = _ci_text()
    assert "sciaudit.evaluator.score" in text
    assert WARMUP_GOLD in text


def test_ci_runs_forbidden_key_scan_on_public_data():
    text = _ci_text()
    assert "forbidden_key_scan" in text
    assert "data_public/" in text


def test_ci_guards_metrics_output():
    text = _ci_text()
    assert "warmup_metrics.json" in text
    assert "invalid_predictions" in text


def test_no_commented_todo_steps_for_existing_modules():
    for line in _ci_text().splitlines():
        stripped = line.strip()
        assert not stripped.startswith("# - run:"), f"commented TODO step: {stripped}"


def test_readme_documents_the_same_chain():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for fragment in CHAIN_FRAGMENTS + (WARMUP_INPUTS, WARMUP_GOLD):
        assert fragment in readme, f"README missing chain fragment: {fragment}"
