"""Тесты рекурсивного скана запрещённых ключей.

Покрывают критерии приёмки ворот утечки:
- собственные публичные деревья репозитория сканируются чисто (код 0);
- подложенное приватное поле ловится, координата печатается (код 1);
- сканируются и ключи, *и* строковые значения;
- профили не дают падать законным файлам gold и предсказаний, но всё ещё
  ловят внутри них стресс-метаданные, провенанс и сплит;
- список запрещённого покрывает список из мануала (Listing 4).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sciaudit.schemas import FORBIDDEN_INPUT_KEYS, find_forbidden, find_forbidden_keys
from sciaudit.leakage.forbidden_key_scan import (
    DEFAULT_RULES,
    PROFILES,
    ScanError,
    classify,
    load_rules,
    scan,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DATA_PUBLIC = REPO / "data_public"

# Список из самого мануала (Listing 4). Проект вправе его расширять, но не сокращать.
MANUAL_LISTING_4 = [
    "gold", "gold_verdict", "verdict", "supporting_eids", "severity",
    "stress", "stress_type", "is_stress_case", "seed_instance_id",
    "evidence_removal", "scope_expansion", "numeric_perturbation", "distractor_flag",
    "private_rationale", "split", "private_slice",
    "GoldHidden", "AutoStressHidden", "ChallengeHidden",
    "provenance_map", "source_url", "paper_title", "authors", "venue",
    "license_status", "adjudication_note", "TA_note",
]

VALID_INPUT = {
    "schema_version": "track_a_input_v1",
    "paper_id": "P001",
    "instance_id": "sas_8f3kq2m9",
    "claim": {"text": "A claim.", "claim_type": "ablation", "scope": "Judge from evidence."},
    "evidence_pack": [{"eid": "e01", "source_kind": "table", "modality": "table_text",
                       "text": "Table 2: 91.2 vs 90.8."}],
    "allowed_evidence_ids": ["e01"],
}


def _write_jsonl(path, objs):
    path.write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n", encoding="utf-8"
    )


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "sciaudit.leakage.forbidden_key_scan", *map(str, args)],
        capture_output=True, text=True, cwd=REPO,
    )


# --- сам список запрещённого ----------------------------------------------------

@pytest.mark.parametrize("key", MANUAL_LISTING_4)
def test_manual_forbidden_key_is_covered(key):
    assert key in FORBIDDEN_INPUT_KEYS


# --- скан ключей и значений -----------------------------------------------------

def test_finds_nested_forbidden_key():
    hits = find_forbidden({"a": {"b": {"gold": {"verdict": "warranted"}}}})
    assert "$.a.b.gold" in hits


def test_finds_forbidden_string_value():
    hits = find_forbidden({"slice_name": "AutoStressHidden"})
    assert any("AutoStressHidden" in h for h in hits)


def test_finds_forbidden_value_inside_list():
    hits = find_forbidden({"slices": ["public_dev", "ChallengeHidden"]})
    assert any("$.slices[1]" in h for h in hits)


def test_prose_mentioning_a_forbidden_word_is_not_a_leak():
    # только точное совпадение: свободный текст не должен ронять ворота
    hits = find_forbidden({"text": "The authors report a gold standard split of the data."})
    assert hits == []


def test_key_only_variant_ignores_values():
    assert find_forbidden_keys({"slice_name": "AutoStressHidden"}) == []


# --- профили --------------------------------------------------------------------

def test_gold_profile_allows_the_label_but_not_stress_metadata():
    forbidden = PROFILES["gold"]
    gold_row = {"instance_id": "sas_001",
                "gold": {"verdict": "warranted", "supporting_eids": ["e01"], "issue_tags": []}}
    assert find_forbidden(gold_row, forbidden) == []
    assert find_forbidden({"stress_type": "evidence_removal"}, forbidden)
    assert find_forbidden({"provenance_ref": "arxiv:1234"}, forbidden)


def test_prediction_profile_allows_verdict_but_not_gold():
    forbidden = PROFILES["prediction"]
    assert find_forbidden({"verdict": "overclaimed", "confidence": 0.7}, forbidden) == []
    assert find_forbidden({"gold": {"verdict": "overclaimed"}}, forbidden)


def test_input_profile_forbids_everything_including_verdict():
    assert find_forbidden({"verdict": "overclaimed"}, PROFILES["input"])


@pytest.mark.parametrize("name,expected", [
    ("inputs.jsonl", "input"),
    ("anything_else.json", "input"),
    ("gold.jsonl", "gold"),
    ("toy_gold.jsonl", "gold"),
    ("sample_predictions.jsonl", "prediction"),
    ("toy_stress_cases.jsonl", "internal"),
    ("sample_internal_annotation.synthetic.jsonl", "internal"),
])
def test_classify_assigns_expected_profile(tmp_path, name, expected):
    profile, reason = classify(tmp_path / name, tmp_path)
    assert profile == expected
    assert reason


def test_every_exemption_carries_a_reason():
    for glob, profile, reason in DEFAULT_RULES:
        assert profile in PROFILES, glob
        assert reason and reason != "(причина не указана)", glob


# --- скан настоящих деревьев ----------------------------------------------------

def test_repo_public_trees_scan_clean():
    leaks, examined = scan([EXAMPLES, DATA_PUBLIC])
    assert leaks == []
    assert examined > 0


def test_scan_catches_planted_leak(tmp_path):
    bad = dict(VALID_INPUT, gold={"verdict": "overclaimed"})
    _write_jsonl(tmp_path / "inputs.jsonl", [VALID_INPUT, bad])
    leaks, _ = scan([tmp_path])
    # вложенные запрещённые ключи сообщаются по отдельности: $.gold и $.gold.verdict
    assert leaks and all("inputs.jsonl:2" in leak for leak in leaks)
    assert any(leak.endswith("$.gold") for leak in leaks)


def test_scan_recurses_into_subdirectories(tmp_path):
    nested = tmp_path / "public_warmup" / "deeper"
    nested.mkdir(parents=True)
    _write_jsonl(nested / "inputs.jsonl", [dict(VALID_INPUT, stress_type="claim_strengthening")])
    leaks, _ = scan([tmp_path])
    assert leaks


def test_scan_reads_plain_json_files_too(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"warmup": {"provenance_map": "…"}}}), encoding="utf-8"
    )
    leaks, _ = scan([tmp_path])
    assert any("$.splits.warmup.provenance_map" in leak for leak in leaks)


def test_scan_ignores_unsupported_formats(tmp_path):
    (tmp_path / "notes.txt").write_text("stress_type: evidence_removal", encoding="utf-8")
    leaks, examined = scan([tmp_path])
    assert leaks == []
    assert examined == 0


def test_unparseable_file_fails_loudly_rather_than_being_skipped(tmp_path):
    (tmp_path / "inputs.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ScanError):
        scan([tmp_path])


def test_missing_target_is_an_error():
    with pytest.raises(ScanError):
        scan([Path("does/not/exist")])


def test_forced_profile_overrides_classification(tmp_path):
    # gold-файл, просканированный под профилем input, обязан упасть: переопределение
    # существует ровно для того, чтобы более строгий прогон доказал отсутствие метки
    _write_jsonl(tmp_path / "gold.jsonl", [{"instance_id": "sas_001",
                                            "gold": {"verdict": "warranted"}}])
    assert scan([tmp_path])[0] == []
    assert scan([tmp_path], forced_profile="input")[0]


# --- переопределение правил -----------------------------------------------------

def test_rules_override_is_honoured(tmp_path):
    _write_jsonl(tmp_path / "custom.jsonl", [{"gold": {"verdict": "warranted"}}])
    rules_file = tmp_path / "rules.json"
    rules_file.write_text(
        json.dumps([{"glob": "custom.jsonl", "profile": "internal", "reason": "тестовая фикстура"}]),
        encoding="utf-8",
    )
    assert scan([tmp_path])[0]
    assert scan([tmp_path], rules=load_rules(rules_file))[0] == []


def test_rules_override_rejects_unknown_profile(tmp_path):
    rules_file = tmp_path / "rules.json"
    rules_file.write_text(json.dumps([{"glob": "*", "profile": "nonsense"}]), encoding="utf-8")
    with pytest.raises(ScanError):
        load_rules(rules_file)


# --- контракт CLI ---------------------------------------------------------------

def test_cli_exits_zero_on_clean_tree():
    result = _cli(DATA_PUBLIC, EXAMPLES)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_cli_exits_one_and_reports_location_on_leak(tmp_path):
    _write_jsonl(tmp_path / "inputs.jsonl", [dict(VALID_INPUT, gold={"verdict": "overclaimed"})])
    result = _cli(tmp_path)
    assert result.returncode == 1
    assert "LEAK" in result.stdout
    assert "$.gold" in result.stdout


def test_cli_exits_two_on_bad_usage():
    assert _cli("does/not/exist").returncode == 2
