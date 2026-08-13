"""Тесты детерминированного численного/табличного чекера (issue #15).

Покрывают каталог §11.5: прямые числовые утверждения, проценты против
процентных пунктов, среднее, best/worst, ранг, all/any, абсолютный и
относительный прирост, а также обязательный эталонный кейс, возврат eid и
офлайновую работу (без модели и сети).

Отдельный блок в конце — выбор сущности по claim, нулевая база и разбор формы
«A n vs B m»: на этом чекер ошибался, и тесты сторожат, чтобы не вернулось.
"""
from sciaudit.baselines.numeric_checker import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNKNOWN,
    _measures,
    _subject,
    check_claim_numbers,
)


def _pack(*rows):
    return [
        {"eid": f"e{i:02d}", "source_kind": "table_row", "modality": "table_text",
         "text": text}
        for i, text in enumerate(rows, start=1)
    ]


def _statuses(results):
    return {r["status"] for r in results}


def _failed(results):
    return [r for r in results if r["status"] == STATUS_FAILED]


# --- обязательный эталонный кейс (мануал §11.5) ------------------------------

def test_reference_case_outperforms_all_baselines_fails():
    results = check_claim_numbers(
        "Method X outperforms all baselines.",
        _pack("Method X = 91.2, Baseline C = 92.1"),
    )
    failed = _failed(results)
    assert failed, results
    assert any("Baseline C" in r["reason"] for r in failed)
    assert failed[0]["eids"] == ["e01"]
    assert any("92.1" in r["reason"] for r in failed)


# --- прямые числовые утверждения ----------------------------------------------

def test_direct_numeric_mismatch_fails():
    results = check_claim_numbers(
        "Method X achieves 91.0 accuracy on Dataset A.",
        _pack("Dataset A accuracy: Method X = 88.1, Baseline = 86.4."),
    )
    assert _failed(results)
    assert any("91.0" in r["reason"] and "88.1" in r["reason"] for r in _failed(results))


def test_direct_numeric_match_ok():
    results = check_claim_numbers(
        "Method Y achieves 93.4% accuracy on Benchmark D.",
        _pack("Table 1: Benchmark D accuracy - Method Y = 93.4, previous SOTA = 93.0."),
    )
    assert _statuses(results) == {STATUS_OK}


# --- проценты против процентных пунктов ---------------------------------------

def test_percentage_points_match_is_ok():
    results = check_claim_numbers(
        "Method X improves the accuracy by 10 percentage points.",
        _pack("Method X = 88.1, Baseline = 78.1"),
    )
    assert _statuses(results) == {STATUS_OK}


def test_percent_is_not_percentage_points():
    # 10 процентных пунктов от базы 78.1 — это ~12.8%, а не 10%.
    results = check_claim_numbers(
        "Method X improves the accuracy by 10%.",
        _pack("Method X = 88.1, Baseline = 78.1"),
    )
    assert _failed(results)
    assert any("12.8" in r["reason"] for r in _failed(results))
    assert any("percentage points and percent are different" in r["reason"]
               for r in _failed(results))


def test_percent_gain_match_is_ok():
    results = check_claim_numbers(
        "The proposed method reduces training time by 5%.",
        _pack("Proposed method = 76.0, baseline = 80.0"),
    )
    assert _statuses(results) == {STATUS_OK}


# --- среднее --------------------------------------------------------------------

def test_mean_check():
    pack = _pack("Method X = 89.0", "Method X = 91.0")
    ok = check_claim_numbers("The mean accuracy is 90.0.", pack)
    bad = check_claim_numbers("The mean accuracy is 90.5.", pack)
    assert _statuses(ok) == {STATUS_OK}
    assert _failed(bad) and any("90.0" in r["reason"] for r in _failed(bad))


# --- best / worst / ранг ----------------------------------------------------------

def test_highest_check_fails_when_baseline_wins():
    results = check_claim_numbers(
        "Method X achieves the highest accuracy.",
        _pack("Method X = 88.0, Baseline = 89.5"),
    )
    assert _failed(results)
    assert any("89.5" in r["reason"] for r in _failed(results))


def test_highest_check_ok_when_method_wins():
    results = check_claim_numbers(
        "Method X achieves the highest accuracy.",
        _pack("Method X = 88.0, Baseline = 87.0"),
    )
    assert _statuses(results) == {STATUS_OK}


def test_rank_first_check():
    results = check_claim_numbers(
        "Method X ranks first among the compared methods.",
        _pack("Method X = 91.2, Baseline C = 92.1"),
    )
    assert _failed(results)


# --- именованное сравнение ---------------------------------------------------------

def test_comparison_fails_when_target_wins():
    results = check_claim_numbers(
        "Method X outperforms Baseline C.",
        _pack("Method X = 91.2, Baseline C = 92.1"),
    )
    assert _failed(results)
    assert any("92.1" in r["reason"] for r in _failed(results))


def test_comparison_ok_when_method_wins():
    results = check_claim_numbers(
        "Method X outperforms Baseline B.",
        _pack("Method X = 91.2, Baseline B = 90.8"),
    )
    assert _statuses(results) == {STATUS_OK}


# --- all/any ------------------------------------------------------------------------

def test_all_baselines_beaten_is_ok():
    results = check_claim_numbers(
        "Method X outperforms all baselines.",
        _pack("Method X = 91.2, Baseline B = 90.8, Baseline C = 89.9"),
    )
    assert _statuses(results) == {STATUS_OK}


# --- абсолютный прирост ---------------------------------------------------------------

def test_absolute_gain_match_is_ok():
    results = check_claim_numbers(
        "Method X improves accuracy by 2.0 points.",
        _pack("Method X = 91.2, Baseline = 89.2"),
    )
    assert _statuses(results) == {STATUS_OK}


def test_absolute_gain_mismatch_fails():
    results = check_claim_numbers(
        "Method X improves accuracy by 2.0 points.",
        _pack("Method X = 91.2, Baseline = 90.0"),
    )
    assert _failed(results)
    assert any("1.2" in r["reason"] for r in _failed(results))


# --- краевые случаи -----------------------------------------------------------------

def test_no_numbers_never_fails():
    results = check_claim_numbers(
        "The proposed approach improves performance.",
        _pack("We describe the architecture in Section 3."),
    )
    assert not _failed(results)
    assert _statuses(results).issubset({STATUS_UNKNOWN, STATUS_OK})


def test_unknown_when_pack_lacks_comparables():
    results = check_claim_numbers(
        "Method X outperforms all baselines.",
        _pack("Figure 2 shows qualitative examples."),
    )
    assert _statuses(results) == {STATUS_UNKNOWN}


def test_empty_pack_does_not_crash():
    results = check_claim_numbers("Method X achieves 90.0 accuracy.", [])
    assert isinstance(results, list)


def test_normalized_numbers_are_used():
    pack = [
        {
            "eid": "e01", "source_kind": "table_row", "modality": "table_text",
            "text": "See Table 1.",
            "normalized_numbers": [
                {"value": 88.1, "unit": "percent", "context": "Method X accuracy"},
                {"value": 86.4, "unit": "percent", "context": "Baseline accuracy"},
            ],
        }
    ]
    results = check_claim_numbers("Method X achieves 91.0 accuracy.", pack)
    assert _failed(results)
    assert results[0]["eids"] == ["e01"]


# --- регрессии: выбор сущности, нулевая база, форма «vs» -------------------------

def test_subject_is_read_from_the_claim():
    assert _subject("Method X outperforms all baselines.") == "Method X"
    assert _subject("The proposed method reduces training time by 5%.") == "proposed method"
    assert _subject("Nothing is asserted here") == ""


def test_pack_with_two_methods_uses_the_one_the_claim_names():
    """Регрессия: бралось первое подходящее число, а не названная сущность.

    Method X (95.0) действительно выше бейзлайна (80.0) — верный ответ ok.
    Раньше проверка хваталась за Method A (70.0) и выдавала ложное failed.
    """
    results = check_claim_numbers(
        "Method X outperforms all baselines.",
        _pack("Method A = 70.0, Method X = 95.0, Baseline = 80.0"),
    )
    assert _statuses(results) == {STATUS_OK}, results


def test_label_in_both_hint_lists_is_not_compared_with_itself():
    """«Baseline model» содержит и «baseline», и «model».

    Признак бейзлайна сильнее, поэтому сущность не должна сравниваться сама с
    собой: раньше в reason выходило «Baseline model is not below Baseline model».
    """
    results = check_claim_numbers(
        "Method X outperforms all baselines.",
        _pack("Baseline model = 95.0, Method X = 88.0"),
    )
    failed = _failed(results)
    assert failed, results
    reason = failed[0]["reason"]
    assert "Baseline model (95.0)" in reason
    assert "Method X (88.0)" in reason


def test_zero_baseline_is_unknown_not_a_crash():
    """Относительный прирост от нуля не определён — это unknown, а не ZeroDivisionError."""
    results = check_claim_numbers(
        "Method X improves accuracy by 10%.",
        _pack("Method X = 5.0, Baseline = 0"),
    )
    assert _statuses(results) == {STATUS_UNKNOWN}, results
    assert any("zero base" in r["reason"] for r in results)


def test_vs_form_is_parsed():
    """«ours 90.2 vs best baseline 89.5» — частая форма в реальных паках."""
    pack = _pack("Table 3: Dataset 1: ours 90.2 vs best baseline 89.5")
    assert [(m.label, m.value) for m in _measures(pack)] == [
        ("ours", 90.2), ("best baseline", 89.5),
    ]
    assert _statuses(check_claim_numbers("Ours outperforms all baselines.", pack)) == {STATUS_OK}


def test_vs_form_fails_when_the_baseline_wins():
    pack = _pack("Table 3: ours 89.5 vs best baseline 90.2")
    assert _failed(check_claim_numbers("Ours outperforms all baselines.", pack))


def test_prose_numbers_are_not_pulled_into_the_pack():
    """Шаблон требует маркер «vs» именно затем, чтобы не тянуть числа из прозы.

    Свободная пара «метка число» дала бы измерения вроде («severity levels 1 to»,
    5.0), и best/worst начал бы падать на пустом месте.
    """
    for text in (
        "Figure 2 reports robustness results for severity levels 1 to 5; "
        "only severity level 1 is included in the supplied evidence.",
        "Table 3 contains ablation results, but the table values are not included.",
        "Figure 3 compares robustness under corruption severity levels 1-5.",
    ):
        assert _measures(_pack(text)) == [], text


def test_duplicate_numbers_are_not_counted_twice():
    """Один и тот же факт, пойманный двумя шаблонами, не должен удваивать среднее."""
    pack = _pack("Method X = 90.0", "Method X = 90.0")
    assert len(_measures(pack)) == 2  # разные eid — это два независимых измерения
    assert len(_measures(_pack("Method X = 90.0"))) == 1
