"""Тесты аудита пересечения сплитов (мануал §10.4).

Проверяют и то, что модуль ловит, и — что важнее — то, на чём он обязан
молчать: общий evidence pack при разных claim и стресс-пару внутри сплита.
"""
import json
from pathlib import Path

import pytest

from sciaudit.leakage.split_overlap_check import (
    DEFAULT_THRESHOLD,
    OverlapError,
    check,
    find_overlaps,
    normalize,
    similarity,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
WARMUP = REPO / "data_public" / "public_warmup"

PACK = [{"eid": "e01", "source_kind": "table", "modality": "table_text",
         "text": "Table 1: ours 90.2, baseline 89.5."}]


def _instance(instance_id, claim_text):
    return {
        "schema_version": "track_a_input_v1",
        "paper_id": "P001",
        "instance_id": instance_id,
        "claim": {"text": claim_text, "claim_type": "ablation", "scope": "Judge from evidence."},
        "evidence_pack": PACK,
        "allowed_evidence_ids": ["e01"],
    }


def _write(path, objs):
    path.write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n", encoding="utf-8"
    )


# --- вспомогательные функции --------------------------------------------------

def test_normalize_ignores_case_and_whitespace():
    assert normalize("  Method   X  Wins ") == "method x wins"


def test_similarity_is_one_for_the_same_text_written_differently():
    assert similarity("Method X wins.", "  method   x   wins.  ") == 1.0


# --- реальные сплиты репозитория ----------------------------------------------

def test_repo_splits_are_disjoint():
    """examples/ и public_warmup/ не должны пересекаться — иначе warm-up
    перестаёт быть независимым замером."""
    problems, sizes = check([EXAMPLES, WARMUP])
    assert problems == [], problems
    assert all(n > 0 for n in sizes.values())


# --- что модуль обязан ловить --------------------------------------------------

def test_detects_verbatim_claim_across_splits(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a / "inputs.jsonl", [_instance("sas_aaaaaaaa", "Component A is necessary.")])
    _write(b / "inputs.jsonl", [_instance("sas_bbbbbbbb", "Component A is necessary.")])

    problems, _ = check([a, b])
    assert any("cross_split_claim" in p for p in problems)


def test_detects_paraphrase_across_splits(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a / "inputs.jsonl", [_instance("sas_aaaaaaaa", "Component A is necessary for the gains.")])
    _write(b / "inputs.jsonl", [_instance("sas_bbbbbbbb", "Component A is necessary for the gain.")])

    problems, _ = check([a, b])
    assert any("cross_split_claim" in p for p in problems)


def test_detects_instance_id_collision(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a / "inputs.jsonl", [_instance("sas_dupdupid", "Claim one about accuracy.")])
    _write(b / "inputs.jsonl", [_instance("sas_dupdupid", "A completely different statement on latency.")])

    problems, _ = check([a, b])
    assert any("id_collision" in p for p in problems)


def test_detects_verbatim_copy_inside_one_split(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _write(a / "inputs.jsonl", [
        _instance("sas_aaaaaaaa", "Component A is necessary."),
        _instance("sas_bbbbbbbb", "component a is NECESSARY."),
    ])
    problems, _ = check([a])
    assert any("intra_split_duplicate" in p for p in problems)


# --- на чём модуль обязан молчать ----------------------------------------------

def test_shared_evidence_pack_with_different_claims_is_not_overlap(tmp_path):
    """Минимальная пара warranted/overclaimed на одном паке — это контроль."""
    a = tmp_path / "a"
    a.mkdir()
    _write(a / "inputs.jsonl", [
        _instance("sas_bounded01", "Method X outperforms the baseline under Gaussian noise."),
        _instance("sas_unbound01", "The model is robust to all corruption types at every severity."),
    ])
    problems, _ = check([a])
    assert problems == []


def test_stress_pair_inside_one_split_is_not_flagged(tmp_path):
    """seed → усиленный claim почти дословно совпадают, и это норма (§5.4)."""
    a = tmp_path / "a"
    a.mkdir()
    _write(a / "inputs.jsonl", [
        _instance("sas_seedaaaa", "Method X improves accuracy on Dataset A."),
        _instance("sas_stressaa", "Method X improves accuracy on all tested datasets."),
    ])
    problems, _ = check([a])
    assert problems == []


def test_internal_annotation_examples_are_not_a_split(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _write(a / "sample_internal_annotation.synthetic.jsonl",
           [_instance("sas_aaaaaaaa", "Component A is necessary.")])
    _write(a / "inputs.jsonl", [_instance("sas_bbbbbbbb", "Component A is necessary.")])
    problems, sizes = check([a])
    assert problems == []
    assert sizes[str(a)] == 1


def test_gold_and_prediction_files_are_ignored(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _write(a / "gold.jsonl", [{"instance_id": "sas_aaaaaaaa",
                               "gold": {"verdict": "warranted", "supporting_eids": [], "issue_tags": []}}])
    _write(a / "predictions.jsonl", [{"instance_id": "sas_aaaaaaaa", "verdict": "warranted",
                                      "confidence": 0.5, "predicted_eids": [], "issue_tags": []}])
    problems, sizes = check([a])
    assert problems == []
    assert sizes[str(a)] == 0


# --- порог и ошибки ------------------------------------------------------------

def test_threshold_is_honoured(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write(a / "inputs.jsonl", [_instance("sas_aaaaaaaa", "Method X improves accuracy on Dataset A.")])
    _write(b / "inputs.jsonl", [_instance("sas_bbbbbbbb", "Method X improves accuracy on Dataset B.")])

    assert check([a, b], threshold=0.99)[0] == []
    assert check([a, b], threshold=0.80)[0]


def test_default_threshold_clears_the_intentional_minimal_pair():
    """Порог обязан быть выше, чем сходство намеренно похожих инстансов."""
    bounded = "Method X outperforms the baseline under Gaussian noise corruption."
    unbounded = "The model is robust to all common corruption types at every severity level."
    assert similarity(bounded, unbounded) < DEFAULT_THRESHOLD


def test_missing_target_is_an_error():
    with pytest.raises(OverlapError):
        check([Path("does/not/exist")])


def test_broken_json_fails_loudly(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "inputs.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(OverlapError):
        check([a])


def test_find_overlaps_reports_both_ids():
    splits = {
        "left": [{"instance_id": "sas_left0001", "claim": "Component A is necessary.", "where": "l:1"}],
        "right": [{"instance_id": "sas_right001", "claim": "Component A is necessary.", "where": "r:1"}],
    }
    problems = find_overlaps(splits)
    assert len(problems) == 1
    assert "sas_left0001" in problems[0] and "sas_right001" in problems[0]
