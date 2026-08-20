"""Тесты B2 и сравнения B1/B2 (issue #14).

B2 — научный контроль к B1: тот же вызов модели, но без ретрива. Ценность
контроля держится на том, что различие между бейзлайнами ровно одно, поэтому
отдельный блок тестов сторожит именно это.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sciaudit.baselines import b1_bm25_llm, b2_fullpack_llm, compare_baselines, model_audit
from sciaudit.baselines.model_audit import count_fallbacks
from sciaudit.schemas.validate_predictions import validate_prediction_file

REPO = Path(__file__).resolve().parents[1]
WARMUP = REPO / "data_public" / "public_warmup"


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_input(instance_id="sas_b2test01"):
    return {
        "schema_version": "track_a_input_v1",
        "paper_id": "P901",
        "instance_id": instance_id,
        "claim": {"text": "Method X improves accuracy on every dataset.",
                  "claim_type": "baseline_superiority",
                  "scope": "Judge only from the supplied evidence pack."},
        "allowed_evidence_ids": ["e01", "e02", "e03"],
        "evidence_pack": [
            {"eid": "e01", "source_kind": "table_row", "modality": "table_text",
             "text": "Dataset A accuracy: Method X = 91.2, Baseline = 89.7."},
            {"eid": "e02", "source_kind": "caption", "modality": "caption_text",
             "text": "Figure 4 shows qualitative examples."},
            {"eid": "e03", "source_kind": "table_row", "modality": "table_text",
             "text": "Dataset B accuracy: Method X = 84.1, Baseline = 85.0."},
        ],
    }


# --- селектор: ровно в нём вся разница -----------------------------------------

def test_b2_selector_returns_the_whole_pack_in_order():
    pack = make_input()["evidence_pack"]
    assert [e["eid"] for e in b2_fullpack_llm.select_full_pack("claim", pack)] == \
        ["e01", "e02", "e03"]


def test_b1_selector_truncates_where_b2_does_not():
    pack = make_input()["evidence_pack"]
    selected = b1_bm25_llm.make_selector(top_k=1)("Dataset A accuracy", pack)
    assert len(selected) == 1
    assert len(b2_fullpack_llm.select_full_pack("Dataset A accuracy", pack)) == 3


def test_b2_prompt_carries_evidence_that_b1_drops(tmp_path):
    """Смысл контроля: B2 обязан видеть то, что ретрив выбросил."""
    instance = make_input()
    seen = {}

    def capture(label):
        def model_fn(prompt):
            seen[label] = prompt
            return json.dumps({"verdict": "overclaimed", "confidence": 0.5,
                               "predicted_eids": [], "issue_tags": [],
                               "abstain": False, "rationale_short": "x"})
        return model_fn

    model_audit.audit_instance(instance, b1_bm25_llm.make_selector(top_k=1),
                               model_fn=capture("b1"))
    model_audit.audit_instance(instance, b2_fullpack_llm.select_full_pack,
                               model_fn=capture("b2"))

    assert "e03" not in seen["b1"]
    assert "e03" in seen["b2"]


# --- общий слой действительно общий ---------------------------------------------

@pytest.mark.parametrize("name", [
    "normalize_prediction", "safe_prediction", "build_prompt", "call_model",
    "parse_model_json", "count_fallbacks", "run_cli", "build_arg_parser",
])
def test_both_baselines_use_the_same_shared_implementation(name):
    """Регрессия против копипасты: AC #14 требует одну реализацию на двоих."""
    shared = getattr(model_audit, name)
    for module in (b1_bm25_llm, b2_fullpack_llm):
        if hasattr(module, name):
            assert getattr(module, name) is shared, module.__name__


def test_baseline_run_delegates_to_the_shared_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run(input_path, output_path, select_evidence, **kwargs):
        calls.append(select_evidence)
        return []

    monkeypatch.setattr(model_audit, "run", fake_run)
    b1_bm25_llm.run("in.jsonl", "out.jsonl", top_k=2, model_fn=lambda p: "{}")
    b2_fullpack_llm.run("in.jsonl", "out.jsonl", model_fn=lambda p: "{}")
    assert len(calls) == 2


# --- контракт выхода -------------------------------------------------------------

def test_b2_output_is_schema_valid(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input()])

    b2_fullpack_llm.run(
        input_path, output_path,
        model_fn=lambda p: json.dumps({"verdict": "overclaimed", "confidence": 0.7,
                                       "predicted_eids": ["e01", "e03", "ghost"],
                                       "issue_tags": ["claim_stronger_than_evidence",
                                                      "made_up"],
                                       "abstain": False, "rationale_short": "ok"}))

    saved = read_jsonl(output_path)
    assert saved[0]["predicted_eids"] == ["e01", "e03"]
    assert saved[0]["issue_tags"] == ["claim_stronger_than_evidence"]
    assert validate_prediction_file(output_path, input_path) == []


def test_b2_falls_back_and_stays_schema_valid(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input()])

    preds = b2_fullpack_llm.run(input_path, output_path, retries=0,
                                model_fn=lambda p: "not json")
    assert count_fallbacks(preds) == 1
    assert validate_prediction_file(output_path, input_path) == []


def test_b2_cli_entry_point_runs(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input()])

    result = subprocess.run(
        [sys.executable, "-m", "sciaudit.baselines.b2_fullpack_llm",
         "--input", str(input_path), "--output", str(output_path),
         "--model-command", f"{sys.executable} -m sciaudit.baselines.stub_model",
         "--model-name", "stub"],
        capture_output=True, text=True, cwd=REPO)

    assert result.returncode == 0, result.stdout + result.stderr
    assert validate_prediction_file(output_path, input_path) == []


def test_b2_rejects_top_k():
    """У B2 нет бюджета ретрива, и притворяться, что есть, он не должен."""
    result = subprocess.run(
        [sys.executable, "-m", "sciaudit.baselines.b2_fullpack_llm",
         "--input", "x", "--output", "y", "--top-k", "3"],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode != 0
    assert "--top-k" in result.stderr


def test_b2_total_fallback_exits_nonzero(tmp_path):
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_jsonl(input_path, [make_input()])
    code = b2_fullpack_llm.main([
        "--input", str(input_path), "--output", str(output_path),
        "--model-command", "this-command-does-not-exist-12345", "--retries", "0"])
    assert code == 1


# --- заглушка модели --------------------------------------------------------------

def test_stub_is_deterministic_and_parses_the_prompt():
    from sciaudit.baselines import stub_model
    prompt = model_audit.build_prompt("Method X wins on every dataset.",
                                      make_input()["evidence_pack"])
    claim, eids = stub_model.parse_prompt(prompt)
    assert claim == "Method X wins on every dataset."
    assert eids == ["e01", "e02", "e03"]
    assert stub_model.decide(claim, eids) == stub_model.decide(claim, eids)


def test_stub_reacts_to_how_much_evidence_it_sees():
    from sciaudit.baselines import stub_model
    claim = "Method X wins on every dataset."
    assert stub_model.decide(claim, ["e01"])["verdict"] == "insufficient"
    assert stub_model.decide(claim, ["e01", "e02"])["verdict"] == "overclaimed"


# --- харнесс сравнения --------------------------------------------------------------

def test_compare_produces_one_row_per_system(tmp_path):
    rows = compare_baselines.compare(
        WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
        compare_baselines.STUB_COMMAND, compare_baselines.STUB_MODEL_NAME,
        top_ks=(1, 2), workdir=str(tmp_path))
    assert [r["system"] for r in rows] == [
        "B1 (BM25, top-k=1)", "B1 (BM25, top-k=2)", "B2 (весь пак)"]
    assert all(r["fallbacks"] == 0 for r in rows), rows
    assert all(r["total"] == 17 for r in rows)
    # бейзлайны обязаны отвечать на каждый инстанс, иначе evaluator их не скорит
    assert all(r["coverage"] == 1.0 for r in rows)


def test_retrieval_budget_changes_the_result(tmp_path):
    """Если бюджет ретрива ни на что не влияет, контроль бессмысленен."""
    rows = compare_baselines.compare(
        WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
        compare_baselines.STUB_COMMAND, compare_baselines.STUB_MODEL_NAME,
        top_ks=(1, 2), workdir=str(tmp_path))
    by_name = {r["system"]: r for r in rows}
    assert by_name["B1 (BM25, top-k=1)"]["accuracy"] != by_name["B2 (весь пак)"]["accuracy"]


def test_markdown_warns_loudly_when_numbers_come_from_the_stub():
    rows = [{"system": "B2 (весь пак)", "accuracy": 0.4, "macro_f1": 0.2,
             "evidence_f1": 0.8, "sfwr": 0.1, "coverage": 1.0, "augrc": 0.25,
             "answered": 17, "total": 17, "fallbacks": 0}]
    stub_md = compare_baselines.to_markdown(rows, "stub", "in", "gold", is_stub=True)
    real_md = compare_baselines.to_markdown(rows, "real", "in", "gold", is_stub=False)
    assert "не моделью" in stub_md and "ничего не измеряет" in stub_md
    assert "не моделью" not in real_md
    # Разброс между одинаковыми прогонами измерен и оказался больше разрыва
    # между строками таблицы. Без этой оговорки таблицу прочтут как вывод.
    assert "вывод не следует" in real_md and "вывод не следует" in stub_md


def test_markdown_shows_the_command_that_actually_produced_it():
    """Инструкция «как перегенерировать» обязана совпадать с прогоном.

    Иначе она отправляет читателя на другой срез и другой транспорт, и числа
    воспроизвести нельзя.
    """
    rows = [{"system": "B2 (весь пак)", "accuracy": 0.4, "macro_f1": 0.2,
             "evidence_f1": 0.8, "sfwr": 0.1, "coverage": 1.0, "augrc": 0.25,
             "answered": 24, "total": 24, "fallbacks": 0}]
    md = compare_baselines.to_markdown(
        rows, "real", "data_public/public_dev/inputs.jsonl",
        "data_public/public_dev/gold.jsonl", is_stub=False,
        transport="--model-api", top_ks="1 2 3")
    assert "--input data_public/public_dev/inputs.jsonl" in md
    assert "--model-api" in md
    assert "--top-k 1 2 3" in md
    assert "public_warmup" not in md


def test_comparison_can_run_through_the_api_transport(tmp_path):
    """B1 и B2 в сравнении обязаны ходить в модель тем же путём, что и поодиночке.

    Пока харнесс умел только --model-command, сравнение через API было
    невозможно, хотя оба бейзлайна его уже поддерживали.
    """
    calls = []

    def fake_model(prompt):
        calls.append(prompt)
        return ('{"verdict": "insufficient", "confidence": 0.3, '
                '"predicted_eids": [], "issue_tags": [], "abstain": true, '
                '"rationale_short": "заглушка вместо сети"}')

    rows = compare_baselines.compare(
        WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
        model_command=None, model_name="api-model",
        top_ks=(1,), workdir=str(tmp_path), model_fn=fake_model)

    assert len(calls) == 17 * 2          # B1 top-k=1 и B2, по разу на инстанс
    assert [r["system"] for r in rows] == ["B1 (BM25, top-k=1)", "B2 (весь пак)"]
    assert all(r["coverage"] == 0.0 for r in rows)


# --- повторы и разброс --------------------------------------------------------------
#
# Замер разброса (docs/baselines_compared.md) показал, что один прогон сравнения систем не
# обосновывает: победитель менялся между одинаковыми запусками. Поэтому харнесс
# обязан уметь повторы и обязан прямо говорить, когда разброс не позволяет
# назвать победителя.

def _aggregated(name, accuracies):
    """Строка таблицы в том виде, в каком её возвращает агрегация по повторам."""
    import statistics
    row = {"system": name, "repeats": len(accuracies), "values": {},
           "answered": 17, "total": 17, "fallbacks": 0}
    for metric in compare_baselines.METRICS:
        values = accuracies if metric == "accuracy" else [0.5] * len(accuracies)
        row["values"][metric] = list(values)
        row[metric] = statistics.median(values)
    return row


def test_repeats_keep_every_run_not_just_the_last(tmp_path):
    rows = compare_baselines.compare(
        WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
        compare_baselines.STUB_COMMAND, compare_baselines.STUB_MODEL_NAME,
        top_ks=(1,), workdir=str(tmp_path), repeats=3)
    assert all(row["repeats"] == 3 for row in rows)
    assert all(len(row["values"]["accuracy"]) == 3 for row in rows)


def test_the_reported_number_is_the_median_of_the_repeats():
    row = {"system": "B2 (весь пак)", "repeats": 3, "accuracy": 0.5,
           "values": {"accuracy": [0.4, 0.5, 0.9]}}
    assert compare_baselines._cell(row, "accuracy") == "0.500 [0.400–0.900]"


def test_a_single_value_is_printed_without_a_fake_spread():
    row = {"system": "B2 (весь пак)", "repeats": 1, "accuracy": 0.5,
           "values": {"accuracy": [0.5]}}
    assert compare_baselines._cell(row, "accuracy") == "0.500"


def test_an_unstable_order_is_reported_as_unstable():
    """Главное, ради чего делаются повторы: честный отказ назвать победителя."""
    rows = [_aggregated("B1", [0.6, 0.4]), _aggregated("B2", [0.5, 0.7])]
    stable, winners = compare_baselines.stability(rows)
    assert (stable, winners) == (False, ["B1", "B2"])

    md = compare_baselines.to_markdown(rows, "m", "in", "gold", is_stub=False, repeats=2)
    assert "не позволяет назвать победителя" in md
    assert "вывод не следует" not in md   # оговорка про один прогон здесь неуместна


def test_a_stable_order_says_so():
    rows = [_aggregated("B1", [0.6, 0.62]), _aggregated("B2", [0.5, 0.5])]
    stable, winners = compare_baselines.stability(rows)
    assert stable and winners == ["B1", "B1"]
    assert "Порядок систем устойчив" in compare_baselines.to_markdown(
        rows, "m", "in", "gold", is_stub=False, repeats=2)


def test_repeats_below_one_are_refused(tmp_path):
    with pytest.raises(ValueError):
        compare_baselines.compare(
            WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
            compare_baselines.STUB_COMMAND, compare_baselines.STUB_MODEL_NAME,
            top_ks=(1,), workdir=str(tmp_path), repeats=0)


def test_the_harness_can_run_the_whole_lineup(tmp_path):
    """B3 и B4 обязаны попадать в ту же таблицу, что B1 и B2.

    Иначе цену осторожности не с чем сравнить: она измеряется только рядом с
    системой, которая отвечает всегда.
    """
    rows = compare_baselines.compare(
        WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
        compare_baselines.STUB_COMMAND, compare_baselines.STUB_MODEL_NAME,
        top_ks=(1,), workdir=str(tmp_path), systems=("b2", "b3", "b4"),
        votes=2, confidence_threshold=0.9)
    names = [row["system"] for row in rows]
    assert names == ["B2 (весь пак)", "B3 (пак + чекер)", "B4 (отказ, votes=2)"]
    by_name = dict(zip(names, rows))
    # Единственный бейзлайн, которому разрешено не отвечать, — и это видно.
    assert by_name["B2 (весь пак)"]["coverage"] == 1.0
    assert by_name["B4 (отказ, votes=2)"]["coverage"] < 1.0


def test_an_unknown_system_is_refused(tmp_path):
    with pytest.raises(ValueError, match="неизвестный бейзлайн"):
        compare_baselines.compare(
            WARMUP / "inputs.jsonl", WARMUP / "gold.jsonl",
            compare_baselines.STUB_COMMAND, compare_baselines.STUB_MODEL_NAME,
            workdir=str(tmp_path), systems=("b7",))
