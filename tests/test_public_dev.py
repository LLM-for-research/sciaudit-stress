"""Проверки публичного dev-среза, выведенного из настоящих статей.

Срез получен механической проекцией приватных записей аннотации
(`sciaudit.construction.derive_public`), поэтому проверяется и сам инструмент
проекции, и инварианты уже выложенного среза.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from sciaudit.baselines.b0_always_insufficient import run as b0_run
from sciaudit.construction.derive_public import (
    DeriveError,
    derive,
    project_gold,
    project_input,
)
from sciaudit.evaluator.score import score
from sciaudit.schemas import FORBIDDEN_INPUT_KEYS, find_forbidden, read_jsonl
from sciaudit.schemas.validate_inputs import validate_input_file
from sciaudit.schemas.validate_predictions import validate_prediction_file

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "data_public" / "public_dev"
INPUTS = DEV / "inputs.jsonl"
GOLD = DEV / "gold.jsonl"
MANIFEST = DEV / "manifest.json"

VERDICTS = {"warranted", "overclaimed", "contradicted", "insufficient"}
ID_PATTERN = re.compile(r"^sas_[a-z0-9]{8}$")
SEMANTIC_HINTS = ("warrant", "overclaim", "contrad", "insufficient", "stress",
                  "remov", "distract", "abstain", "gold", "leak", "seed", "trap")

INTERNAL_RECORD = {
    "schema_version": "internal_annotation_v1",
    "instance_id": "sas_derive01",
    "paper": {"paper_id": "P001", "provenance_ref": "P001",
              "domain": "robotics", "year_bucket": "2024_2026"},
    "claim": {"text": "Method X wins on every dataset.", "claim_type": "robustness",
              "claim_strength": "strong", "scope": "Judge only from the evidence.",
              "source_location": "section_4"},
    "evidence_pack": [
        {"eid": "e01", "source_kind": "table_row", "modality": "table_text",
         "text": "Dataset A: Method X = 91.2, Baseline = 89.7.",
         "source_ref": "table_2", "is_distractor": False,
         "normalized_numbers": [{"value": 91.2, "unit": "percent", "context": "Method X"}]},
        {"eid": "e02", "source_kind": "caption", "modality": "caption_text",
         "text": "Figure 4 shows qualitative examples.",
         "source_ref": "figure_4", "is_distractor": True},
    ],
    "stress": {"is_stress_case": True, "stress_type": "scope_expansion",
               "seed_instance_id": "sas_derive00"},
    "gold": {"verdict": "overclaimed", "supporting_eids": ["e01"],
             "issue_tags": ["claim_stronger_than_evidence"], "severity": "moderate",
             "private_rationale": "ПРИВАТНО: одна таблица не покрывает every dataset."},
    "review": {"validation_level": "team_verified", "reviewer": "r3based"},
    "split": "public_dev",
}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


# --- инструмент проекции ------------------------------------------------------------

def test_projection_strips_every_private_field():
    public = project_input(INTERNAL_RECORD)

    assert find_forbidden(public, FORBIDDEN_INPUT_KEYS) == []
    assert set(public) == {"schema_version", "paper_id", "instance_id", "claim",
                           "evidence_pack", "allowed_evidence_ids"}
    assert set(public["claim"]) == {"text", "claim_type", "scope"}
    for unit in public["evidence_pack"]:
        assert "source_ref" not in unit and "is_distractor" not in unit


def test_projection_keeps_claim_and_evidence_verbatim():
    """Проекция вычёркивает поля, но не переписывает тексты."""
    public = project_input(INTERNAL_RECORD)
    assert public["claim"]["text"] == INTERNAL_RECORD["claim"]["text"]
    assert [u["text"] for u in public["evidence_pack"]] == \
        [u["text"] for u in INTERNAL_RECORD["evidence_pack"]]
    assert public["evidence_pack"][0]["normalized_numbers"] == \
        INTERNAL_RECORD["evidence_pack"][0]["normalized_numbers"]


def test_allowed_ids_cover_the_whole_pack():
    public = project_input(INTERNAL_RECORD)
    assert public["allowed_evidence_ids"] == [u["eid"] for u in public["evidence_pack"]]


def test_public_gold_carries_the_label_but_not_the_rationale():
    gold = project_gold(INTERNAL_RECORD)
    assert gold["gold"]["verdict"] == "overclaimed"
    assert "severity" not in gold["gold"]
    assert "private_rationale" not in gold["gold"]
    assert "ПРИВАТНО" not in json.dumps(gold, ensure_ascii=False)


def test_duplicate_instance_ids_are_refused(tmp_path):
    path = tmp_path / "internal.jsonl"
    _write(path, [INTERNAL_RECORD, INTERNAL_RECORD])
    with pytest.raises(DeriveError, match="дубликат"):
        derive([path])


def test_record_without_instance_id_is_refused(tmp_path):
    broken = {k: v for k, v in INTERNAL_RECORD.items() if k != "instance_id"}
    path = tmp_path / "internal.jsonl"
    _write(path, [broken])
    with pytest.raises(DeriveError, match="instance_id"):
        derive([path])


def test_derived_files_pass_the_input_validator(tmp_path):
    path = tmp_path / "internal.jsonl"
    _write(path, [INTERNAL_RECORD])
    inputs, golds = derive([path])
    _write(tmp_path / "inputs.jsonl", inputs)
    assert validate_input_file(tmp_path / "inputs.jsonl") == []
    assert len(golds) == 1


# --- выложенный срез ------------------------------------------------------------------

def test_split_exists_and_gold_covers_every_input():
    inputs = {obj["instance_id"]: obj for _, obj in read_jsonl(str(INPUTS))}
    golds = {obj["instance_id"]: obj["gold"] for _, obj in read_jsonl(str(GOLD))}

    assert len(inputs) >= 20
    assert set(inputs) == set(golds)


def test_inputs_validate_and_leak_nothing():
    assert validate_input_file(INPUTS) == []


def test_instance_ids_are_non_semantic():
    for _, obj in read_jsonl(str(INPUTS)):
        instance_id = obj["instance_id"]
        assert ID_PATTERN.match(instance_id), instance_id
        assert not any(hint in instance_id for hint in SEMANTIC_HINTS), instance_id


def test_all_four_verdicts_present_and_none_dominates():
    counts = Counter(obj["gold"]["verdict"] for _, obj in read_jsonl(str(GOLD)))
    total = sum(counts.values())
    assert set(counts) == VERDICTS
    assert max(counts.values()) / total <= 0.5, counts


def test_every_gold_field_is_from_the_allowed_vocabulary():
    allowed_tags = set(json.loads(
        (REPO / "schemas" / "prediction.schema.json").read_text("utf-8")
    )["properties"]["issue_tags"]["items"]["enum"])
    inputs = {obj["instance_id"]: obj for _, obj in read_jsonl(str(INPUTS))}

    for _, row in read_jsonl(str(GOLD)):
        gold = row["gold"]
        assert gold["verdict"] in VERDICTS
        assert set(gold["issue_tags"]) <= allowed_tags, gold
        allowed_eids = set(inputs[row["instance_id"]]["allowed_evidence_ids"])
        assert set(gold["supporting_eids"]) <= allowed_eids, row["instance_id"]


def test_instances_come_from_real_papers_not_placeholders():
    """Настоящие статьи нумеруются с P001; P901-P915 зарезервированы под синтетику."""
    paper_ids = {obj["paper_id"] for _, obj in read_jsonl(str(INPUTS))}
    assert paper_ids
    assert all(not pid.startswith("P9") for pid in paper_ids), paper_ids


def test_manifest_names_no_labels():
    """Провенанс не должен делать вердикт угадываемым по метаданным."""
    text = MANIFEST.read_text(encoding="utf-8").lower()
    for word in ("warranted", "overclaimed", "contradicted", "insufficient",
                 "severity", "private_rationale", "claim_strengthening",
                 "evidence_removal", "scope_expansion"):
        assert word not in text, word


def test_manifest_lists_every_instance():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {row["instance_id"] for row in manifest["instances"]}
    actual = {obj["instance_id"] for _, obj in read_jsonl(str(INPUTS))}
    assert listed == actual
    assert manifest["dataset_shape"]["instances"] == len(actual)


def test_every_instance_was_checked_by_a_human():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert all(row["checked_by_human"] for row in manifest["instances"])


def test_evidence_packs_are_richer_than_the_synthetic_warmup():
    """Смысл настоящих статей в том, что паки крупнее и разнообразнее."""
    packs = [len(obj["evidence_pack"]) for _, obj in read_jsonl(str(INPUTS))]
    assert min(packs) >= 2
    assert sum(packs) / len(packs) > 2.5


def test_readiness_loop_runs_on_the_dev_split(tmp_path):
    """B0 и evaluator обязаны отработать на срезе без ошибок."""
    predictions = tmp_path / "b0.jsonl"
    b0_run(str(INPUTS), str(predictions))

    assert validate_prediction_file(predictions, INPUTS) == []

    metrics = score(predictions, GOLD)
    assert metrics["submission"]["scoreable"] is True
    assert metrics["counts"]["missing_predictions"] == 0
    assert metrics["counts"]["gold_instances"] == len(list(read_jsonl(str(INPUTS))))


def test_published_comparison_carries_the_variance_caveat():
    """Опубликованная таблица без оговорки о разбросе читается как вывод.

    Два одинаковых прогона сравнения дали противоположный ответ на вопрос
    «помогает ли ретрив», поэтому оговорка — часть результата, а не украшение.
    """
    note = (REPO / "docs" / "b1_vs_b2.md").read_text(encoding="utf-8")
    assert "вывод не следует" in note
    assert "--model-api" in note, "инструкция должна называть транспорт прогона"
    assert "public_warmup" not in note, "инструкция ссылается не на тот срез"
