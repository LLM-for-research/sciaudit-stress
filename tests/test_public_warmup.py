"""Public warm-up dataset checks (issue #9).

Guards the acceptance criteria of the public warm-up slice:
- inputs.jsonl exists with >= 10 instances; gold.jsonl covers every instance;
- the input validator passes (schema + no private-field leakage);
- all four verdicts are present and none exceeds 50%;
- instance IDs are non-semantic (sas_ + random suffix, no verdict/stress hints);
- verdict is not trivially correlated with claim wording or claim type
  (claim-only probe, staff manual §10.3);
- B0 runs on the warm-up inputs and the evaluator scores it without errors.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from sciaudit.evaluator.score import score
from sciaudit.schemas import find_forbidden_keys, read_jsonl
from sciaudit.schemas.validate_inputs import validate_input_file
from sciaudit.schemas.validate_predictions import validate_prediction_file
from sciaudit.baselines.b0_always_insufficient import run as b0_run

REPO = Path(__file__).resolve().parents[1]
WARMUP = REPO / "data_public" / "public_warmup"
INPUTS = WARMUP / "inputs.jsonl"
GOLD = WARMUP / "gold.jsonl"
MANIFEST = WARMUP / "manifest.json"

VERDICTS = {"warranted", "overclaimed", "contradicted", "insufficient"}
ID_PATTERN = re.compile(r"^sas_[a-z0-9]{8}$")
SEMANTIC_HINTS = (
    "warrant", "overclaim", "contrad", "insufficient", "stress", "remov",
    "distract", "abstain", "gold", "leak", "seed", "trap",
)
HEDGE_WORDS = ("may", "suggests", "reported", "in most", "appears", "can improve")


def _golds():
    return {obj["instance_id"]: obj["gold"] for _, obj in read_jsonl(str(GOLD))}


def _inputs():
    return {obj["instance_id"]: obj for _, obj in read_jsonl(str(INPUTS))}


# --- files exist and are consistent -----------------------------------------

def test_warmup_files_exist():
    assert INPUTS.exists(), "public_warmup/inputs.jsonl missing"
    assert GOLD.exists(), "public_warmup/gold.jsonl missing"
    assert MANIFEST.exists()


def test_at_least_ten_instances():
    assert len(_inputs()) >= 10


def test_gold_covers_exactly_the_input_instances():
    assert set(_golds()) == set(_inputs()), (
        "gold.jsonl must have exactly one record per instance_id"
    )


def test_inputs_pass_validator():
    assert validate_input_file(str(INPUTS)) == []


def test_inputs_have_no_private_fields():
    for instance_id, obj in _inputs().items():
        hits = find_forbidden_keys(obj)
        assert not hits, f"{instance_id} leaks private fields: {hits}"


# --- verdict balance -----------------------------------------------------------

def test_all_four_verdicts_present_and_none_above_half():
    counts = Counter(v["verdict"] for v in _golds().values())
    assert set(counts) == VERDICTS
    assert max(counts.values()) / sum(counts.values()) <= 0.5, counts


# --- non-semantic IDs ----------------------------------------------------------

def test_instance_ids_are_non_semantic():
    for instance_id in _inputs():
        assert ID_PATTERN.match(instance_id), f"non-conforming id: {instance_id}"
        assert not any(hint in instance_id for hint in SEMANTIC_HINTS)


def test_paper_ids_are_abstracted():
    for obj in _inputs().values():
        paper_id = obj["paper_id"]
        assert re.match(r"^P\d{3}$", paper_id), f"non-abstracted paper_id: {paper_id}"


# --- no trivial verdict correlation (claim-only probe) --------------------------

def test_no_claim_type_predicts_a_single_verdict():
    by_type = {}
    for instance_id, obj in _inputs().items():
        by_type.setdefault(obj["claim"]["claim_type"], set()).add(
            _golds()[instance_id]["verdict"]
        )
    for claim_type, verdicts in by_type.items():
        if len(verdicts) < 2:
            continue  # single instance: nothing to leak
        assert len(verdicts) >= 2, f"claim_type {claim_type} always -> {verdicts}"


def test_warranted_does_not_correlate_with_soft_wording():
    hedged = {i for i, o in _inputs().items()
              if any(w in o["claim"]["text"].lower() for w in HEDGE_WORDS)}
    non_hedged = set(_inputs()) - hedged

    hedged_verdicts = {_golds()[i]["verdict"] for i in hedged}
    non_hedged_verdicts = {_golds()[i]["verdict"] for i in non_hedged}

    # both subsets must be verdict-mixed, so the label is not guessable from phrasing
    assert len(hedged_verdicts) >= 2, f"hedged wording only -> {hedged_verdicts}"
    assert len(non_hedged_verdicts) >= 3, f"strong wording only -> {non_hedged_verdicts}"


# --- end-to-end: B0 + evaluator on the warm-up slice -----------------------------

def test_b0_runs_on_warmup_and_evaluator_scores_it(tmp_path):
    out = tmp_path / "b0_warmup.jsonl"
    n = b0_run(str(INPUTS), str(out))
    assert n == len(_inputs())

    assert validate_prediction_file(str(out), str(INPUTS)) == []

    metrics = score(str(out), str(GOLD))
    assert metrics["counts"]["gold_instances"] == len(_inputs())
    assert metrics["counts"]["invalid_predictions"] == 0
    assert metrics["counts"]["missing_predictions"] == 0
    assert "accuracy" in metrics["verdict"]
    assert "macro_f1" in metrics["verdict"]

    # B0 always predicts insufficient: accuracy equals the gold insufficient share,
    # and B0 can never commit a severe false warrant.
    insufficient_share = (
        sum(1 for v in _golds().values() if v["verdict"] == "insufficient")
        / len(_golds())
    )
    assert metrics["verdict"]["accuracy"] == pytest.approx(insufficient_share)
    assert metrics["severe_false_warrant_rate"]["count"] == 0
