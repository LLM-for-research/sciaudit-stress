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
    polarity,
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


# --- направление метрики --------------------------------------------------------
#
# До этого чекер считал, что больше всегда лучше. На метриках ошибки (RMSE,
# секунды, падение в пунктах) он выдавал failed на верном claim, то есть
# уверенно врал — хуже, чем если бы молчал. Половина реальных таблиц такие,
# поэтому вторую статью пилота ему нельзя было показывать вообще.

def _row(text, eid="e01", **extra):
    return [{"eid": eid, "source_kind": "table_row", "modality": "table_text",
             "text": text, **extra}]


def test_explicit_marker_flips_the_comparison():
    out = check_claim_numbers(
        "Method X outperforms the Baseline.",
        _row("Out-of-distribution RMSE (lower is better): Method X = 21.76, Baseline = 99.73."))
    assert [r["status"] for r in out] == ["ok"]
    assert "lower-is-better" in out[0]["reason"]


def test_metric_name_alone_is_enough():
    """Пометки может не быть — «latency in seconds» само по себе задаёт направление."""
    out = check_claim_numbers(
        "Method X outperforms the Baseline.",
        _row("Inference latency in seconds: Method X = 12.4, Baseline = 40.1."))
    assert [r["status"] for r in out] == ["ok"]


def test_higher_is_better_stays_the_default():
    out = check_claim_numbers(
        "Method X outperforms the Baseline.",
        _row("Accuracy: Method X = 91.2, Baseline = 88.1."))
    assert [r["status"] for r in out] == ["ok"]


def test_a_wrong_claim_on_an_error_metric_still_fails():
    """Разворот направления не должен превращать чекер в поддакивающий."""
    out = check_claim_numbers(
        "Method X outperforms all baselines.",
        _row("Test RMSE (lower is better): Method X = 44.0, Baseline A = 21.76."))
    assert [r["status"] for r in out] == ["failed"]


def test_unit_of_a_normalized_number_carries_direction():
    out = check_claim_numbers(
        "Method X outperforms the Baseline.",
        _row("Method X = 12.4, Baseline = 40.1.",
              normalized_numbers=[
                  {"context": "Method X", "value": 12.4, "unit": "seconds"},
                  {"context": "Baseline", "value": 40.1, "unit": "seconds"},
              ]))
    assert [r["status"] for r in out] == ["ok"]


def test_opposite_directions_are_unknown_not_failed():
    """Точность против времени сравнивать нельзя — и выдумывать вердикт тоже."""
    out = check_claim_numbers(
        "Method X outperforms the Baseline.",
        [{"eid": "e01", "text": "Accuracy (higher is better): Method X = 91.2."},
         {"eid": "e02", "text": "Latency in seconds (lower is better): Baseline = 40.1."}])
    assert [r["status"] for r in out] == ["unknown"]
    assert "opposite directions" in out[0]["reason"]


def test_gain_on_an_error_metric_counts_the_reduction():
    out = check_claim_numbers(
        "Method X reduces error by 22.5 points over the Baseline.",
        _row("RMSE (lower is better): Method X = 21.5, Baseline = 44.0."))
    assert [r["status"] for r in out] == ["ok"]


def test_gain_on_an_error_metric_catches_a_wrong_number():
    out = check_claim_numbers(
        "Method X reduces error by 40.0 points over the Baseline.",
        _row("RMSE (lower is better): Method X = 21.5, Baseline = 44.0."))
    assert [r["status"] for r in out] == ["failed"]
    assert "22.5" in out[0]["reason"]


def test_best_on_an_error_metric_means_lowest():
    out = check_claim_numbers(
        "Method X is the best model in the table.",
        _row("Validation loss (lower is better): Method X = 0.21, Model B = 0.44."))
    assert [r["status"] for r in out] == ["ok"]


def test_literal_superlative_is_not_flipped_by_the_metric():
    """«Наибольшее» — про величину, а не про качество; направление его не трогает."""
    out = check_claim_numbers(
        "Method X has the highest loss in the table.",
        _row("Validation loss (lower is better): Method X = 0.21, Model B = 0.44."))
    assert [r["status"] for r in out] == ["failed"]


def test_polarity_helper_prefers_the_author_marker_over_the_metric_name():
    """Строка «Success rate drop (higher is better)» содержит и «drop», и пометку."""
    assert polarity("Success rate drop (higher is better)") is False
    assert polarity("Success rate drop") is True
    assert polarity("Accuracy") is None


# --- ранжирующие утверждения ----------------------------------------------------
#
# Здесь чекер опаснее всего: он вправе перебивать модель в B3, поэтому уверенно
# неверный ответ хуже молчания. Оба дефекта ниже нашлись на живом инстансе
# sas_r095qb9w, где gold — insufficient, а чекер выдавал contradicted.

def test_ordinal_place_is_not_read_as_the_superlative():
    """«Второй по величине» — не «наибольший»."""
    pack = _row("Average drop (lower is better): language = 27.84, "
                "background = 19.04, light = 23.47.")
    out = check_claim_numbers(
        "Language perturbation causes the second-smallest average drop.", pack)
    assert [r["status"] for r in out] == [STATUS_FAILED]
    assert "ranks 3" in out[0]["reason"] and "not 2" in out[0]["reason"]


def test_correct_ordinal_place_passes():
    pack = _row("Average drop (lower is better): language = 27.84, "
                "background = 19.04, light = 23.47.")
    out = check_claim_numbers(
        "Language perturbation causes the third-smallest average drop.", pack)
    assert [r["status"] for r in out] == [STATUS_OK]


def test_numeric_ordinal_form_is_understood():
    pack = _row("Accuracy: Method X = 88.0, Model B = 91.0, Model C = 95.0.")
    out = check_claim_numbers("Method X is the 3rd highest in the table.", pack)
    assert [r["status"] for r in out] == [STATUS_OK]


def test_ranking_claim_about_an_absent_entity_is_unknown():
    """Не найдя сущность claim'а, чекер обязан молчать, а не брать первую попавшуюся.

    Пак перечисляет модели, claim — про метрику, которой в паке нет вовсе.
    Догадка давала уверенное failed про сущность, о которой claim не говорил.
    """
    pack = _row("Accuracy: Model A = 53.5, Model B = 17.6, Model C = 24.8.")
    out = check_claim_numbers(
        "Among the seven perturbation families, language perturbation causes the "
        "second-smallest average drop in success rate.", pack)
    assert [r["status"] for r in out] == [STATUS_UNKNOWN]
    assert "not named in the evidence pack" in out[0]["reason"]


# --- ряды таблицы ---------------------------------------------------------------
#
# Реальные паки устроены строками: одно условие — много чисел по моделям. Claim
# при этом сравнивает строки между собой, а не отдельные ячейки. Пока чекер
# видел только ячейки, содержательный ответ выходил на 2 инстансах из 24.

def _families():
    return [
        {"eid": "e02", "text": "Language perturbation, absolute drop per model: "
                               "Model A = 53.5, Model B = 17.6, Model C = 24.8, "
                               "Model D = 16.6, Model E = 35.4, Model F = 24.5, "
                               "Model G = 22.8, Model H = 37.5, Model I = 25.7, "
                               "Model J = 20.0."},
        {"eid": "e03", "text": "Light perturbation, absolute drop per model: "
                               "Model A = 68.4, Model B = 8.4, Model C = 18.5, "
                               "Model D = 4.9, Model E = 9.2, Model F = 12.3, "
                               "Model G = 42.2, Model H = 35.4, Model I = 26.2, "
                               "Model J = 9.2."},
        {"eid": "e04", "text": "Background perturbation, absolute drop per model: "
                               "Model A = 41.7, Model B = 3.8, Model C = 1.7, "
                               "Model D = 6.6, Model E = 12.8, Model F = 12.3, "
                               "Model G = 29.3, Model H = 62.0, Model I = 14.3, "
                               "Model J = 5.9."},
    ]


def test_row_becomes_one_measure_named_by_its_header():
    from sciaudit.baselines.numeric_checker import _series
    got = {m.label: m.value for m in _series(_families())}
    assert got == {"Language perturbation": 27.84,
                   "Light perturbation": 23.47,
                   "Background perturbation": 19.04}
    assert all(m.lower_is_better for m in _series(_families())), "«drop» — метрика ошибки"


def test_ranking_over_rows_catches_a_wrong_place():
    """Тот самый дефект статьи: заявлено второе место, на деле третье."""
    out = check_claim_numbers(
        "Among the seven perturbation families, language perturbation causes the "
        "second-smallest average drop in success rate.", _families())
    assert [r["status"] for r in out] == [STATUS_FAILED]
    assert "ranks 3" in out[0]["reason"]


def test_cutting_the_pack_makes_the_same_claim_uncheckable():
    """Минимальная пара seed → evidence_removal.

    Тот же claim, урезанный пак. Ранжировать одно семейство не с чем, и
    честный ответ — «не с чем сравнивать», а не «противоречит».
    """
    out = check_claim_numbers(
        "Among the seven perturbation families, language perturbation causes the "
        "second-smallest average drop in success rate.", _families()[:1])
    assert [r["status"] for r in out] == [STATUS_UNKNOWN]


def test_rows_do_not_leak_into_the_mean_check():
    """Среднее по паку считается по числам, а не по средним строк."""
    out = check_claim_numbers("The mean accuracy is 90.0.",
                              _row("Accuracy: Method X = 89.0, Model B = 91.0."))
    assert [r["status"] for r in out] == [STATUS_OK]


def test_ranking_still_works_when_the_entity_is_named_in_the_claim():
    """Ужесточение не должно отключить проверку там, где сущность в паке есть."""
    out = check_claim_numbers(
        "Method X is the best model in the table.",
        _row("Validation loss (lower is better): Method X = 0.21, Model B = 0.44."))
    assert [r["status"] for r in out] == [STATUS_OK]


def test_entity_name_needs_word_boundaries():
    """«Model B» не должна находиться внутри «model by».

    Нашлось на живом инстансе: claim «surpassing the next-best model by 37.2
    percentage points» назначал субъектом случайную модель из таблицы и сверял
    с ней чужое число. Ответ совпадал с gold по случайности.
    """
    pack = _row("Camera-viewpoint perturbation, success rate per model: "
                "Model A = 0.8, Model B = 56.4, Model C = 10.4.")
    out = check_claim_numbers(
        "The proposed model reaches 92.8 success, surpassing the next-best "
        "model by 37.2 percentage points.", pack)
    # Подходят две проверки — прямое число и «next-best»; обе обязаны молчать.
    assert _statuses(out) == {STATUS_UNKNOWN}


def test_a_row_name_in_the_claim_is_a_condition_not_a_subject():
    """Имя строки без слова об агрегате — это условие эксперимента.

    Иначе результат одной модели сверялся бы со средним по всем моделям.
    """
    pack = _row("Camera-viewpoint perturbation, success rate per model: "
                "Model A = 0.8, Model B = 56.4, Model C = 10.4.")
    out = check_claim_numbers(
        "Model A reaches 0.8 under camera-viewpoint perturbation.", pack)
    assert [r["status"] for r in out] == [STATUS_OK]


def test_an_aggregate_claim_is_checked_against_the_row_mean():
    out = check_claim_numbers(
        "The average drop caused by language perturbation is 25.3 points.",
        _families())
    assert [r["status"] for r in out] == [STATUS_FAILED]
    assert "27.84" in out[0]["reason"]
