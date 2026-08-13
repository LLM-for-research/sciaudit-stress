"""Проверки публичного warm-up датасета (issue #9).

Сторожат критерии приёмки публичного warm-up среза:
- inputs.jsonl существует и содержит >= 10 инстансов; gold.jsonl покрывает все;
- валидатор входов проходит (схема + отсутствие утечки приватных полей);
- присутствуют все четыре вердикта, и ни один не занимает больше 50%;
- instance_id несемантические (sas_ + случайный суффикс, без намёков на вердикт и стресс);
- вердикт тривиально не коррелирует с формулировкой claim и его типом
  (проба только по claim, мануал §10.3);
- B0 отрабатывает на warm-up входах, и evaluator считает по ним метрики без ошибок.
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
# Слова-кванторы всеобщности, которыми обычно злоупотребляют преувеличенные
# claim (проба только по claim, мануал §10.3): вердикт не должен читаться из
# одного лишь наличия или отсутствия квантора.
UNIV_QUANTIFIER = re.compile(r"\b(all|every|consistently|always|any)\b", re.I)

# Любое из этих слов в метаданных warm-up сделало бы gold-метку конкретных
# инстансов угадываемой (страховка ревью: провенанс обязан оставаться без
# меток, даже если сам gold.jsonl публичен в учебных целях).
LABEL_VOCABULARY = (
    "warranted", "overclaimed", "contradicted", "insufficient",
    "claim_strengthening", "scope_expansion", "evidence_removal",
    "distractor_evidence", "numeric_mismatch", "table_caption_mismatch",
    "missing_baseline", "weak_ablation", "non_comparable_baseline",
    "severity", "private_rationale",
)


def _golds():
    return {obj["instance_id"]: obj["gold"] for _, obj in read_jsonl(str(GOLD))}


def _inputs():
    return {obj["instance_id"]: obj for _, obj in read_jsonl(str(INPUTS))}


# --- файлы существуют и согласованы ------------------------------------------

def test_warmup_files_exist():
    assert INPUTS.exists(), "нет public_warmup/inputs.jsonl"
    assert GOLD.exists(), "нет public_warmup/gold.jsonl"
    assert MANIFEST.exists()


def test_at_least_ten_instances():
    assert len(_inputs()) >= 10


def test_gold_covers_exactly_the_input_instances():
    assert set(_golds()) == set(_inputs()), (
        "в gold.jsonl должна быть ровно одна запись на instance_id"
    )


def test_inputs_pass_validator():
    assert validate_input_file(str(INPUTS)) == []


def test_inputs_have_no_private_fields():
    for instance_id, obj in _inputs().items():
        hits = find_forbidden_keys(obj)
        assert not hits, f"{instance_id} сливает приватные поля: {hits}"


# --- баланс вердиктов ------------------------------------------------------------

def test_all_four_verdicts_present_and_none_above_half():
    counts = Counter(v["verdict"] for v in _golds().values())
    assert set(counts) == VERDICTS
    assert max(counts.values()) / sum(counts.values()) <= 0.5, counts


# --- несемантические ID -----------------------------------------------------------

def test_instance_ids_are_non_semantic():
    for instance_id in _inputs():
        assert ID_PATTERN.match(instance_id), f"id не соответствует формату: {instance_id}"
        assert not any(hint in instance_id for hint in SEMANTIC_HINTS)


def test_paper_ids_are_abstracted():
    for obj in _inputs().values():
        paper_id = obj["paper_id"]
        assert re.match(r"^P\d{3}$", paper_id), f"paper_id не абстрагирован: {paper_id}"


def test_warmup_metadata_contains_no_label_vocabulary():
    for name in ("manifest.json", "README.md"):
        text = (WARMUP / name).read_text(encoding="utf-8").lower()
        # Законные упоминания, случайно содержащие подстроки из словаря:
        # имя модуля B0 встречается в документированных командах запуска.
        text = text.replace("b0_always_insufficient", "")
        for word in LABEL_VOCABULARY:
            assert word not in text, f"{name} содержит словарь меток: '{word}'"


# --- нет тривиальной корреляции с вердиктом (проба только по claim) -------------

def test_no_claim_type_predicts_a_single_verdict():
    by_type = {}
    type_counts = Counter()
    for instance_id, obj in _inputs().items():
        claim_type = obj["claim"]["claim_type"]
        type_counts[claim_type] += 1
        by_type.setdefault(claim_type, set()).add(_golds()[instance_id]["verdict"])
    for claim_type, verdicts in by_type.items():
        if type_counts[claim_type] >= 2:
            assert len(verdicts) >= 2, (
                f"claim_type {claim_type} (n={type_counts[claim_type]}) "
                f"всегда даёт -> {verdicts}"
            )


def test_verdict_not_predictable_from_quantifier_wording():
    quantifier = {}
    for instance_id, obj in _inputs().items():
        has_q = bool(UNIV_QUANTIFIER.search(obj["claim"]["text"]))
        quantifier.setdefault(has_q, set()).add(_golds()[instance_id]["verdict"])

    # И подмножество с квантором, и подмножество без него обязаны быть смешанными
    # по вердикту, а warranted обязан встречаться среди claim с квантором (классическая
    # лазейка: есть квантор => overclaimed, нет квантора => warranted).
    for has_q, verdicts in quantifier.items():
        assert len(verdicts) >= 3, f"квантор={has_q} даёт только -> {verdicts}"
    assert "warranted" in quantifier.get(True, set())


# --- сквозной прогон: B0 + evaluator на warm-up срезе ----------------------------

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

    # B0 всегда предсказывает insufficient: accuracy равна доле insufficient в gold,
    # и severe false warrant B0 совершить не может.
    insufficient_share = (
        sum(1 for v in _golds().values() if v["verdict"] == "insufficient")
        / len(_golds())
    )
    assert metrics["verdict"]["accuracy"] == pytest.approx(insufficient_share)
    # B0 никогда не отказывается от ответа, поэтому SFWR без отказов покрывает все
    # предсказания; warranted B0 не предсказывает никогда, значит severe false warrant
    # невозможен.
    assert metrics["severe_false_warrant_rate_non_abstained"]["count"] == 0
