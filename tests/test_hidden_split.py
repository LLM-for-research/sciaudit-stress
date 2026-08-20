"""Скрытый контур: нарезка (§10.4) и репетиция скрытой оценки (§8.1, §10.5).

Скрытый сплит — это не «данные, которые мы не показали», а данные, которые
**нельзя вывести** из показанного. Отсюда два предмета проверки.

Первый: границу держат группы, а не инстансы. Стресс-вариант отличается от
своего seed одним контролируемым изменением, поэтому вариант в скрытом срезе
при публичном seed — это публичная подсказка, и такая нарезка обязана считаться
сломанной.

Второй: в студенческую половину не уезжает gold. Эту ошибку нельзя заметить по
метрикам — они от неё только улучшатся, — поэтому она ловится здесь.
"""
import json
from pathlib import Path

import pytest

from sciaudit.construction import make_hidden_split as splitter
from sciaudit.evaluator import hidden_dry_run


def _record(instance_id, paper="P001", verdict="warranted", seed=None):
    stress = ({"is_stress_case": True, "stress_type": "claim_strengthening",
               "seed_instance_id": seed} if seed else {"is_stress_case": False})
    return {
        "schema_version": "internal_annotation_v1",
        "instance_id": instance_id,
        "paper": {"paper_id": paper, "provenance_ref": paper},
        "claim": {"text": f"claim {instance_id}", "claim_type": "numerical_performance",
                  "scope": "dataset", "claim_strength": "strong"},
        "evidence_pack": [{"eid": "e01", "source_kind": "table_row",
                           "modality": "table_text", "text": "Accuracy: X = 91.2."}],
        "stress": stress,
        "gold": {"verdict": verdict, "supporting_eids": ["e01"], "issue_tags": [],
                 "severity": "moderate", "private_rationale": "не для участников"},
        "review": {"validation_level": "team_verified", "reviewer": "r3based"},
        "split": "public_dev",
    }


def _ids(n, prefix="sas_a"):
    return [f"{prefix}{i:07d}" for i in range(n)]


def _corpus(papers=3, per_paper=6):
    records = []
    for paper_index in range(papers):
        paper = f"P{paper_index + 1:03d}"
        ids = _ids(per_paper, prefix=f"sas_{paper_index}")
        for position, instance_id in enumerate(ids):
            # каждый третий — стресс-вариант предыдущего
            seed = ids[position - 1] if position % 3 == 2 else None
            records.append(_record(instance_id, paper=paper, seed=seed,
                                   verdict=("warranted", "overclaimed",
                                            "contradicted", "insufficient")[position % 4]))
    return records


def _write_internal(tmp_path, records):
    path = tmp_path / "internal.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")
    return path


# --- границы держат группы ----------------------------------------------------------

def test_a_stress_variant_never_leaves_its_seed_behind():
    """Главный инвариант §10.4: иначе скрытый инстанс подсказан публичным."""
    records = _corpus()
    per_split, problems, _ = splitter.split_records(
        records, "seed", splitter.DEFAULT_RATIOS)

    where = {record["instance_id"]: name
             for name, members in per_split.items() for record in members}
    for record in records:
        seed = (record.get("stress") or {}).get("seed_instance_id")
        if seed:
            assert where[record["instance_id"]] == where[seed], record["instance_id"]
    assert problems == []


def test_paper_mode_keeps_a_paper_whole():
    per_split, problems, paper_splits = splitter.split_records(
        _corpus(), "paper", splitter.DEFAULT_RATIOS)
    assert problems == []
    assert all(len(splits) == 1 for splits in paper_splits.values())
    for members in per_split.values():
        assert len({splitter.paper_key(r) for r in members}) == 1


def test_seed_mode_admits_that_papers_cross_the_border():
    """Слабее §10.4 — и обязано быть названо, а не подразумеваться."""
    _, _, paper_splits = splitter.split_records(
        _corpus(), "seed", splitter.DEFAULT_RATIOS)
    assert any(len(splits) > 1 for splits in paper_splits.values())


def test_a_requested_but_empty_split_is_a_failure():
    """Две статьи на три группы не делятся, и молчать об этом нельзя."""
    records = _corpus(papers=2, per_paper=5)
    _, problems, _ = splitter.split_records(records, "paper", splitter.DEFAULT_RATIOS)
    assert any("заказан, но пуст" in problem for problem in problems), problems


def test_the_slicing_is_deterministic():
    first, _, _ = splitter.split_records(_corpus(), "seed", splitter.DEFAULT_RATIOS)
    second, _, _ = splitter.split_records(_corpus(), "seed", splitter.DEFAULT_RATIOS)
    assert ({k: [r["instance_id"] for r in v] for k, v in first.items()}
            == {k: [r["instance_id"] for r in v] for k, v in second.items()})


def test_the_sizes_follow_the_requested_shares():
    per_split, _, _ = splitter.split_records(
        _corpus(papers=6, per_paper=6), "paper",
        {splitter.PUBLIC_SPLIT: 0.5, splitter.HIDDEN_VAL: 0.3, splitter.HIDDEN_TEST: 0.2})
    sizes = {name: len(members) for name, members in per_split.items()}
    assert sizes[splitter.PUBLIC_SPLIT] > sizes[splitter.HIDDEN_VAL]
    assert sizes[splitter.HIDDEN_VAL] >= sizes[splitter.HIDDEN_TEST]


def test_an_unknown_grouping_is_refused():
    with pytest.raises(ValueError, match="paper или seed"):
        splitter.build_groups(_corpus(), "instance")


# --- что уезжает участникам ---------------------------------------------------------

def test_hidden_students_get_inputs_and_nothing_else(tmp_path):
    per_split, _, _ = splitter.split_records(_corpus(), "seed", splitter.DEFAULT_RATIOS)
    student, staff = tmp_path / "student", tmp_path / "staff"
    for name, members in per_split.items():
        splitter.write_split(name, members, student, staff)

    assert not (student / splitter.HIDDEN_VAL / "gold.jsonl").exists()
    assert not (student / splitter.HIDDEN_TEST / "gold.jsonl").exists()
    assert (staff / splitter.HIDDEN_VAL / "gold.jsonl").exists()
    # публичный срез — единственный, чей gold живёт на студенческой стороне
    assert (student / splitter.PUBLIC_SPLIT / "gold.jsonl").exists()


def test_student_inputs_carry_no_private_fields(tmp_path):
    per_split, _, _ = splitter.split_records(_corpus(), "seed", splitter.DEFAULT_RATIOS)
    student, staff = tmp_path / "student", tmp_path / "staff"
    splitter.write_split(splitter.HIDDEN_VAL, per_split[splitter.HIDDEN_VAL],
                         student, staff)

    text = (student / splitter.HIDDEN_VAL / "inputs.jsonl").read_text(encoding="utf-8")
    for forbidden in ("gold", "stress", "review", "private_rationale", "severity"):
        assert forbidden not in text, forbidden
    assert hidden_dry_run.check_student_half(student, splitter.HIDDEN_VAL) == []


def test_gold_smuggled_into_the_student_half_is_caught(tmp_path):
    """Ошибка, которую нельзя увидеть по метрикам: от неё они только вырастут."""
    per_split, _, _ = splitter.split_records(_corpus(), "seed", splitter.DEFAULT_RATIOS)
    student, staff = tmp_path / "student", tmp_path / "staff"
    splitter.write_split(splitter.HIDDEN_VAL, per_split[splitter.HIDDEN_VAL],
                         student, staff)

    (student / splitter.HIDDEN_VAL / "gold.jsonl").write_text("{}\n", encoding="utf-8")
    problems = hidden_dry_run.check_student_half(student, splitter.HIDDEN_VAL)
    assert any("gold.jsonl" in problem for problem in problems), problems


def test_a_private_field_inside_an_input_row_is_caught(tmp_path):
    per_split, _, _ = splitter.split_records(_corpus(), "seed", splitter.DEFAULT_RATIOS)
    student, staff = tmp_path / "student", tmp_path / "staff"
    splitter.write_split(splitter.HIDDEN_VAL, per_split[splitter.HIDDEN_VAL],
                         student, staff)

    path = student / splitter.HIDDEN_VAL / "inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["gold"] = {"verdict": "warranted"}
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")

    problems = hidden_dry_run.check_student_half(student, splitter.HIDDEN_VAL)
    assert problems


# --- репетиция целиком --------------------------------------------------------------

def test_the_dry_run_goes_from_slicing_to_metrics(tmp_path):
    directory = tmp_path / "in"
    directory.mkdir()
    internal = _write_internal(directory, _corpus())

    problems, report = hidden_dry_run.dry_run([internal], tmp_path / "work", by="seed")
    assert problems == []
    assert report["split"] == splitter.HIDDEN_VAL
    assert report["instances"] > 0
    assert report["coverage"] == 1.0          # B0 отвечает всегда
    assert "accuracy" in report


def test_the_dry_run_reports_when_there_is_nothing_to_rehearse(tmp_path):
    directory = tmp_path / "in"
    directory.mkdir()
    internal = _write_internal(directory, _corpus(papers=1, per_paper=4))

    problems, _ = hidden_dry_run.dry_run(
        [internal], tmp_path / "work", by="paper",
        ratios={splitter.PUBLIC_SPLIT: 0.5, splitter.HIDDEN_VAL: 0.5,
                splitter.HIDDEN_TEST: 0.0})
    assert any("не получился" in problem or "пуст" in problem for problem in problems)


def test_the_cli_runs_on_the_real_private_annotations():
    """Репетиция обязана проходить на тех данных, что есть сегодня."""
    repo = Path(__file__).resolve().parents[1]
    internal = sorted(str(p) for p in (repo / "data_paper_derived").glob(
        "*.internal_annotation.jsonl"))
    assert internal, "приватные аннотации не найдены"
    assert hidden_dry_run.main(["--internal", *internal, "--by", "seed"]) == 0
