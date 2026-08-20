"""Пробы на утечку метки: §10.2 (метаданные), §10.3 (один claim), §5.5 (ID).

Проба на утечку — инструмент с необычным требованием: она обязана **находить**
подложенную утечку и обязана **молчать** на чистых данных. Проверяется и то, и
другое: гейт, который не ловит, бесполезен, а гейт, который срабатывает на шуме,
через неделю выключают.

Отдельно сторожится главный инвариант §10.2 и §10.3: пробы не должны видеть
того, что положено видеть системе. Метаданная проба не смотрит на текст, а
claim-проба — на evidence. Если это разъедется, проба перестанет отвечать на
свой вопрос и начнёт мерить качество аудита.
"""
import json

import pytest

from sciaudit.leakage import claim_only_probe, id_randomness_check, metadata_probe
from sciaudit.leakage import probe_common

VERDICTS = ("warranted", "overclaimed", "contradicted", "insufficient")


def _instance(instance_id, claim="Method X improves accuracy.", units=1,
              claim_type="numerical_performance", paper="P001", unit_text="e" * 50):
    return {
        "instance_id": instance_id,
        "paper_id": paper,
        "claim": {"text": claim, "claim_type": claim_type, "scope": "dataset"},
        "evidence_pack": [{"eid": f"e{i:02d}", "modality": "text",
                           "source_kind": "paragraph", "text": unit_text}
                          for i in range(1, units + 1)],
        "allowed_evidence_ids": [f"e{i:02d}" for i in range(1, units + 1)],
    }


def _write(tmp_path, pairs):
    """Положить (инстанс, вердикт) в пару файлов input/gold и вернуть пути."""
    inputs = tmp_path / "inputs.jsonl"
    gold = tmp_path / "gold.jsonl"
    inputs.write_text("\n".join(json.dumps(i) for i, _ in pairs) + "\n", encoding="utf-8")
    gold.write_text("\n".join(
        json.dumps({"instance_id": i["instance_id"],
                    "gold": {"verdict": v, "supporting_eids": [], "issue_tags": []}})
        for i, v in pairs) + "\n", encoding="utf-8")
    return inputs, gold


def _ids(n):
    """Детерминированные ID нужной формы, не несущие смысла."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    out = []
    for i in range(n):
        tail = "".join(alphabet[(i * 7 + k * 13) % len(alphabet)] for k in range(8))
        out.append(f"sas_{tail}")
    return out


# --- общая машинерия ----------------------------------------------------------------

def test_baselines_describe_the_two_trivial_strategies():
    labels = ["a", "a", "a", "b"]
    assert probe_common.majority_accuracy(labels) == pytest.approx(0.75)
    assert probe_common.stratified_accuracy(labels) == pytest.approx(0.625)


def test_leave_one_out_never_lets_the_classifier_see_its_own_answer():
    seen = []

    def fit_predict(train_rows, train_labels, test_row):
        seen.append((len(train_rows), test_row))
        return train_labels[0]

    rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    probe_common.leave_one_out_accuracy(fit_predict, rows, ["a", "b", "c"])
    assert [size for size, _ in seen] == [2, 2, 2]
    assert [row for _, row in seen] == rows


def test_a_probe_with_no_signal_gets_an_unremarkable_p_value():
    rows = [{"x": float(i % 2)} for i in range(20)]
    labels = ["a", "b"] * 10  # признак и метка независимы

    def constant(train_rows, train_labels, test_row):
        return "a"

    _, p_value, _ = probe_common.permutation_test(constant, rows, labels,
                                                  permutations=50)
    assert p_value > probe_common.ALPHA_SUSPECT
    assert probe_common.verdict_of(p_value) == probe_common.VERDICT_CLEAN


def test_verdict_thresholds():
    assert probe_common.verdict_of(0.001) == probe_common.VERDICT_LEAK
    assert probe_common.verdict_of(0.10) == probe_common.VERDICT_SUSPECT
    assert probe_common.verdict_of(0.50) == probe_common.VERDICT_CLEAN


def test_a_missing_gold_verdict_is_an_error_not_a_silent_skip(tmp_path):
    inputs, gold = _write(tmp_path, [(_instance(_ids(1)[0]), "warranted")])
    gold.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="нет gold-вердикта"):
        probe_common.read_labeled(inputs, gold)


# --- §10.2: проба на метаданных -----------------------------------------------------

def test_the_metadata_probe_never_sees_any_text():
    """Иначе она перестанет отвечать на свой вопрос и станет слабым аудитором."""
    instance = _instance("sas_abcd1234", claim="Уникальная строка claim",
                         unit_text="Уникальная строка evidence")
    features = metadata_probe.features_of(instance)
    blob = " ".join(str(key) for key in features)
    assert "Уникальная строка claim" not in blob
    assert "Уникальная строка evidence" not in blob
    assert features["evidence_count"] == 1.0


def test_a_planted_metadata_leak_is_found(tmp_path):
    """Число единиц evidence жёстко задаёт вердикт — проба обязана это увидеть."""
    pairs = []
    for index, instance_id in enumerate(_ids(24)):
        verdict = VERDICTS[index % 2]
        units = 2 if verdict == "warranted" else 9
        pairs.append((_instance(instance_id, units=units), verdict))
    inputs, gold = _write(tmp_path, pairs)

    assert metadata_probe.main(["--inputs", str(inputs), "--gold", str(gold),
                                "--permutations", "100"]) == 1


def test_the_probe_names_the_field_that_leaks(tmp_path):
    pairs = []
    for index, instance_id in enumerate(_ids(24)):
        verdict = VERDICTS[index % 2]
        units = 2 if verdict == "warranted" else 9
        pairs.append((_instance(instance_id, units=units), verdict))

    rows = [metadata_probe.features_of(instance) for instance, _ in pairs]
    labels = [verdict for _, verdict in pairs]
    tree = metadata_probe.fit_tree(rows, labels)
    assert tree["name"] in ("evidence_count", "allowed_eids_count",
                            "evidence_len_total")


def test_clean_metadata_stays_quiet(tmp_path):
    pairs = [(_instance(instance_id), VERDICTS[index % 4])
             for index, instance_id in enumerate(_ids(24))]
    inputs, gold = _write(tmp_path, pairs)
    assert metadata_probe.main(["--inputs", str(inputs), "--gold", str(gold),
                                "--permutations", "100"]) == 0


# --- §10.3: проба на одном claim ----------------------------------------------------

def test_the_claim_probe_never_sees_the_evidence():
    instance = _instance("sas_abcd1234", claim="method improves accuracy",
                         unit_text="уникальный evidence")
    assert claim_only_probe.claim_tokens(instance) == ["method", "improves", "accuracy"]


def test_a_planted_wording_leak_is_found(tmp_path):
    """Слово «always» встречается только у overclaimed — ровно случай из §10.3."""
    pairs = []
    for index, instance_id in enumerate(_ids(24)):
        verdict = "overclaimed" if index % 2 else "warranted"
        claim = ("The method always wins on every dataset." if verdict == "overclaimed"
                 else "The method reaches 91.2 on the benchmark.")
        pairs.append((_instance(instance_id, claim=claim), verdict))
    inputs, gold = _write(tmp_path, pairs)

    assert claim_only_probe.main(["--inputs", str(inputs), "--gold", str(gold),
                                  "--permutations", "100"]) == 1


def test_the_report_lists_the_words_to_rebalance():
    documents = [["always", "wins"], ["always", "everywhere"],
                 ["reaches", "91"], ["reaches", "88"]]
    labels = ["overclaimed", "overclaimed", "warranted", "warranted"]
    lines = "\n".join(claim_only_probe.indicative_tokens(documents, labels))
    assert "always" in lines and "reaches" in lines


def test_the_same_wording_across_classes_is_not_a_leak(tmp_path):
    """Сильная формулировка сама по себе не метка — это и есть требуемый баланс."""
    pairs = []
    for index, instance_id in enumerate(_ids(24)):
        claim = "The method always wins on every dataset."
        pairs.append((_instance(instance_id, claim=claim), VERDICTS[index % 4]))
    inputs, gold = _write(tmp_path, pairs)
    assert claim_only_probe.main(["--inputs", str(inputs), "--gold", str(gold),
                                  "--permutations", "100"]) == 0


# --- §5.5: идентификаторы -----------------------------------------------------------

def test_the_manual_own_bad_examples_are_rejected():
    bad = ["sas_seed_00045", "sas_scope_stress_001", "autostress_014",
           "goldhidden_022", "P017_scope_expansion"]
    problems = id_randomness_check.check_shapes(bad)
    for instance_id in bad:
        assert any(instance_id in problem for problem in problems), instance_id


def test_good_ids_pass():
    assert id_randomness_check.check_shapes(_ids(10)) == []


def test_a_repeated_id_is_a_violation():
    duplicated = _ids(3) + _ids(1)
    assert any("встречается" in problem
               for problem in id_randomness_check.check_shapes(duplicated))


def test_labels_clumped_by_id_order_are_caught():
    """ID, выданные по ходу разметки, сортируются в блоки одинаковых меток."""
    pairs = [(f"sas_{i:08d}".replace("sas_0", "sas_a"), VERDICTS[i // 6])
             for i in range(24)]
    _, p_value, _ = id_randomness_check.check_order(pairs, permutations=500)
    assert p_value < id_randomness_check.ALPHA_ORDER_FAIL


def test_random_ids_do_not_trip_the_order_check():
    pairs = list(zip(_ids(24), [VERDICTS[i % 4] for i in range(24)]))
    _, p_value, _ = id_randomness_check.check_order(pairs, permutations=500)
    assert p_value > id_randomness_check.ALPHA_ORDER_FAIL


def test_the_cli_fails_on_semantic_ids(tmp_path):
    pairs = [(_instance("sas_scope_stress_001"), "warranted"),
             (_instance("sas_seed_00045"), "overclaimed")]
    inputs, _ = _write(tmp_path, pairs)
    assert id_randomness_check.main(["--inputs", str(inputs)]) == 1


def test_the_cli_passes_on_random_ids(tmp_path):
    pairs = [(_instance(instance_id), VERDICTS[index % 4])
             for index, instance_id in enumerate(_ids(12))]
    inputs, _ = _write(tmp_path, pairs)
    assert id_randomness_check.main(["--inputs", str(inputs)]) == 0
