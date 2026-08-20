#!/usr/bin/env python3
"""Детерминированный численный/табличный чекер для Track A (мануал §11.5).

Офлайновый компонент без зависимостей: сверяет числа из claim с числами из
evidence pack. Модель не дёргается никогда — это строительный блок для
бейзлайна B3 и для инструментов проверки человеком.

Реализованные проверки (из каталога §11.5):

1. прямое числовое утверждение против evidence («achieves 91.0» при
   заявленных 88.1);
2. именованное сравнение («Method X outperforms Baseline B»);
3. all/any («outperforms all baselines») против каждого бейзлайна;
4. best/worst и ранг («highest», «best», «ranks first»);
5. mean/average против среднего по числам пака;
6. абсолютный прирост m − b (пункты, процентные пункты);
7. относительный прирост (m − b) / b в процентах — вместе с различением
   процентов и процентных пунктов (прирост в 10 п.п. не равен приросту на
   10%, типовая ошибка из delegation-лога).

Каждая проверка выдаёт запись::

    {"claim_fragment", "status", "reason", "eids"}

где ``status`` — "ok" (числа сходятся), "failed" (найдено числовое
противоречие) или "unknown" (в паке не с чем сравнивать).

**О ком claim, решает сам claim.** Сущность берётся из формулировки
утверждения и сопоставляется с метками измерений; словарь подсказок
(«method», «baseline», …) — только запасной вариант, когда названную сущность
в паке не нашли. Без этого пак с двумя методами разбирался бы по первому
попавшемуся числу, и верное утверждение получало бы «failed».

Эталонный кейс (§11.5): claim «Method X outperforms all baselines», evidence
«Method X = 91.2, Baseline C = 92.1» → status "failed" с eid строки таблицы.

**Направление метрики берётся из evidence, а не предполагается.** Половина
реальных таблиц — величины ошибки, времени и стоимости, где лучше меньше:
RMSE, секунды на эпоху, падение в пунктах. Проверка «больше значит лучше»
выдавала на таком паке ``failed`` при верном claim, то есть уверенно врала.
Направление ищется по порядку: явная пометка автора («lower is better», «↓»),
затем имя метрики, затем поле ``unit`` у нормализованного числа. Если
направление из evidence не следует, действует «больше — лучше».

Если у двух сравниваемых величин направления известны и различны — например
claim сопоставляет точность со временем — результат ``unknown``, а не
``failed``: величины несопоставимы, и честный ответ здесь «не с чем
сравнивать», а не выдуманное противоречие.
"""
from __future__ import annotations

import re

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"

#: Явная пометка направления метрики в тексте единицы evidence. Автор таблицы
#: пишет её ровно затем, чтобы читатель не гадал, поэтому она сильнее всех
#: остальных признаков.
_LOWER_MARKER = re.compile(
    r"lower\s+(?:is|the)\s+better|less\s+is\s+better|lower[\s=-]+better|\(\s*↓\s*\)|↓",
    re.I,
)
_HIGHER_MARKER = re.compile(
    r"higher\s+(?:is|the)\s+better|more\s+is\s+better|higher[\s=-]+better|\(\s*↑\s*\)|↑",
    re.I,
)

#: Метрики, у которых меньше — лучше. Список намеренно про величины ошибки,
#: стоимости и времени: именно они составляют половину реальных таблиц и
#: именно на них проверка «больше значит лучше» выдаёт failed на верном claim.
_LOWER_METRICS = re.compile(
    r"\b("
    r"rmse|mse|mae|mape|nll|wer|cer|fid|kid|ece|brier|perplexity|ppl|regret|"
    r"error|errors|loss|losses|latency|runtime|inference\s+time|training\s+time|"
    r"wall[-\s]?clock|seconds?|secs?|millisecond\w*|ms\b|minutes?|hours?|"
    r"cost|drop|degradation|decline|violation\w*|failure\s+rate|overhead"
    r")\b",
    re.I,
)

#: Сигнал, что направление у сравниваемых величин разное и сравнивать их
#: арифметически нельзя.
_CONFLICT = object()

# Измерения вида «Method X = 91.2» и «Method Y: 93.4».
_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9 .\-']{0,40}?)\s*[=:]\s*(\d+(?:\.\d+)?)")
# Форма сравнения без знака равенства: «ours 90.2 vs best baseline 89.5».
# Намеренно требуются обе метки: в такой конструкции числа заведомо
# сопоставимы. Свободный шаблон «метка число» затянул бы в пак мусор вроде
# «Table 3» или «severity levels 1 to 5» и породил бы ложные срабатывания.
_VS_PAIR = re.compile(
    r"([A-Za-z][A-Za-z0-9 .\-']{0,30}?)\s+(\d+(?:\.\d+)?)\s+vs\.?\s+"
    r"([A-Za-z][A-Za-z0-9 .\-']{0,30}?)\s+(\d+(?:\.\d+)?)",
    re.I,
)

_METHOD_HINTS = (
    "method", "ours", "proposed", "our model", "full model",
    "our approach", "model",
)
_BASELINE_HINTS = ("baseline", "sota", "previous", "competitor", "without", "standard")

# Сущность, о которой идёт claim: то, что стоит перед глаголом утверждения.
_SUBJECT = re.compile(
    r"^\W*(?:the\s+)?(.{0,60}?)\s+"
    r"(?:outperform|achiev|reach|attain|report|improv|increas|reduc|rank|obtain|"
    r"deliver|cut)\w*\b",
    re.I,
)

#: Прямое числовое утверждение. Форма «… is 25.3 points» требует единицы
#: измерения после числа: без неё «is 7» ловилось бы в любом предложении.
_DIRECT = re.compile(
    r"(?:achieves|reports|reaches|attains|accuracy of|score of|latency of|"
    r"reduces\s+\w+\s+to|improves\s+\w+\s+to)\s+(\d+(?:\.\d+)?)"
    r"|(?:\bis|\bwas|\bequals)\s+(\d+(?:\.\d+)?)\s*"
    r"(?:percentage\s+points?|points?|percent|%)",
    re.I,
)
_COMPARE = re.compile(r"outperform\w*\s+(?:the\s+)?([A-Z][A-Za-z0-9 \-']*)", re.I)
_ALL_ANY = re.compile(
    r"outperform\w*\s+(all|every|any)\s+(?:the\s+)?(?:compared\s+)?baselines?", re.I
)
#: «best» и «ranks first» — про качество, поэтому у метрики ошибки лучший
#: результат наименьший. «highest» и «largest» — про величину буквально, и
#: направление метрики их не переворачивает: claim про наибольшее значение
#: остаётся claim'ом про наибольшее значение.
_BEST = re.compile(r"(best|ranks?\s+first|top[-\s]?1|state[-\s]of[-\s]the[-\s]art)", re.I)
_BEST_LITERAL = re.compile(r"(highest|largest|lowest|smallest)", re.I)

#: Порядковое место: «second smallest», «third-best», «2nd highest». Без этого
#: «второй по величине» читалось как «наибольший», и верный claim о втором месте
#: получал уверенное failed — худший вид ошибки для инструмента, который вправе
#: перебивать модель.
_ORDINAL = re.compile(
    r"\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"(\d+)(?:st|nd|rd|th))[-\s]+(?=\w*(?:smallest|largest|highest|lowest|best|worst))",
    re.I,
)
_ORDINAL_WORDS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
                  "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}

#: Заголовок строки таблицы: «Language perturbation, absolute drop per model: …».
#: Реальные паки устроены рядами — одно условие, много чисел по моделям, — и
#: claim обычно сравнивает не отдельные числа, а ряды между собой. Без этого
#: чекер видел десять чисел там, где утверждение говорит про одно среднее.
#: Заголовок — всё до первого двоеточия, без знака равенства внутри, и за
#: двоеточием должно идти не число: иначе «Method Y: 93.4» разбиралось бы как
#: ряд с заголовком «Method Y». Имя ряда — часть заголовка до первой запятой:
#: в claim ссылаются на «Language perturbation», а не на всю подпись строки.
#: Слово об агрегате: без него имя строки в claim — условие, а не субъект.
_AGGREGATE = re.compile(r"\b(average|mean|overall|aggregate|across\s+(?:all\s+)?models)\b", re.I)

_SERIES_HEAD = re.compile(r"^\s*([^:=]{3,80}?)\s*:\s*(?=\D)")
_MEAN = re.compile(r"(?:mean|average)\s+(?:accuracy|score|latency|performance)?\s*"
                   r"(?:of\s*)?(?:is\s+)?(\d+(?:\.\d+)?)", re.I)
#: Улучшение метрики ошибки описывают глаголами убывания: «reduces RMSE by 12
#: points». Без них проверка прироста просто не подходила к половине claim'ов —
#: тот же перекос «больше значит лучше», только в словаре глаголов.
_ABS_GAIN = re.compile(
    r"(?:improve\w*|increase\w*|gain\w*|reduc\w*|decreas\w*|lower\w*|cut\w*|"
    r"shrink\w*)\s+.{0,60}?\bby\s+(\d+(?:\.\d+)?)\s+"
    r"(?:percentage\s+)?points?", re.I,
)
_PCT_GAIN = re.compile(
    r"(?:improve\w*|increase\w*|reduc\w*)\s+.{0,60}?\bby\s+(\d+(?:\.\d+)?)\s*%", re.I
)

_TOL = 1e-9


class Measure:
    """Одно число, извлечённое из единицы evidence.

    ``lower_is_better`` — направление метрики: True (меньше — лучше), False
    (больше — лучше) или None (из evidence не следует). None и False ведут
    себя одинаково при сравнении, но различаются при конфликте направлений:
    неизвестное направление уступает известному, а два известных и разных
    делают сравнение невозможным.
    """

    __slots__ = ("label", "value", "eid", "lower_is_better", "kind")

    def __init__(self, label: str, value: float, eid: str, lower_is_better=None,
                 kind="point"):
        self.label = label
        self.value = value
        self.eid = eid
        self.lower_is_better = lower_is_better
        #: "point" — число из пака как есть; "series" — среднее по строке
        #: таблицы, названное её заголовком. Ранжировать их вместе нельзя.
        self.kind = kind


def polarity(*texts) -> bool | None:
    """Меньше ли лучше для метрики, названной в этих текстах.

    Порядок признаков: явная пометка автора, затем имя метрики. Явная пометка
    сильнее, потому что строка «Success rate drop (lower is better)» содержит
    и «drop», и осмысленное указание, и спорить с автором тут не о чем.
    """
    for text in texts:
        text = str(text or "")
        if _LOWER_MARKER.search(text):
            return True
        if _HIGHER_MARKER.search(text):
            return False
    for text in texts:
        if _LOWER_METRICS.search(str(text or "")):
            return True
    return None


def _direction(measures):
    """Общее направление для набора измерений.

    Возвращает bool или ``_CONFLICT``, если у измерений направления известны и
    различны: сравнивать секунды с процентами арифметически нельзя, и честный
    ответ на такой claim — «не с чем сравнивать», а не выдуманный вердикт.
    """
    flags = {m.lower_is_better for m in measures if m.lower_is_better is not None}
    if len(flags) > 1:
        return _CONFLICT
    return flags.pop() if flags else False


def _measures(evidence_pack: list[dict]) -> list[Measure]:
    """Разобрать числа из evidence pack: пары «метка = значение» и «A n vs B m»."""
    out: list[Measure] = []
    seen: set[tuple[str, float, str]] = set()

    def add(label: str, value: float, eid: str, *context) -> None:
        label = label.strip()
        key = (label.lower(), value, eid)
        direction = polarity(*context, label)
        if key not in seen:
            seen.add(key)
            out.append(Measure(label, value, eid, direction))
            return
        # То же число, но из другого источника. Одно и то же измерение обычно
        # приходит дважды: разобранным из текста и объявленным в
        # normalized_numbers с полем unit. Второе несёт направление метрики,
        # первое — нет, и молча отбросить его значит потерять единственный
        # надёжный признак.
        if direction is None:
            return
        for existing in out:
            if (existing.label.lower(), existing.value, existing.eid) == key:
                if existing.lower_is_better is None:
                    existing.lower_is_better = direction
                return

    for unit in evidence_pack or []:
        if not isinstance(unit, dict):
            continue
        eid = str(unit.get("eid", "?"))
        text = str(unit.get("text", ""))
        # Направление метрики объявляется на всю единицу evidence: пометка
        # «(lower is better)» стоит в заголовке строки таблицы, а не у каждого
        # числа. Поэтому текст единицы идёт в контекст каждого измерения.
        for label, value in _PAIR.findall(text):
            add(label, float(value), eid, text)
        for left_label, left_value, right_label, right_value in _VS_PAIR.findall(text):
            add(left_label, float(left_value), eid, text)
            add(right_label, float(right_value), eid, text)
        for nb in unit.get("normalized_numbers", []) or []:
            if isinstance(nb, dict) and isinstance(nb.get("value"), (int, float)):
                # У нормализованного числа единица измерения объявлена явно —
                # это самый надёжный источник направления из всех.
                add(str(nb.get("context", "")), float(nb["value"]), eid,
                    nb.get("unit", ""), text)
    return out


def _series(evidence_pack: list[dict]) -> list[Measure]:
    """Ряды: строка таблицы сворачивается в среднее, названное её заголовком.

    Отдельно от :func:`_measures`, а не вместе с ним: подмешивание средних к
    обычным числам испортило бы проверку среднего по паку и сравнение метода с
    бейзлайном. Ряды нужны ровно одной проверке — о месте в таблице.
    """
    out: list[Measure] = []
    for unit in evidence_pack or []:
        if not isinstance(unit, dict):
            continue
        text = str(unit.get("text", ""))
        head = _SERIES_HEAD.match(text)
        if not head:
            continue
        values = [float(v) for _, v in _PAIR.findall(text)]
        if len(values) < 2:
            continue
        out.append(Measure(
            head.group(1).split(",")[0].strip(),
            round(sum(values) / len(values), 6),
            str(unit.get("eid", "?")),
            polarity(text, head.group(1)),
            kind="series",
        ))
    return out


def _subject(claim: str) -> str:
    """Сущность, о которой claim: подлежащее перед глаголом утверждения."""
    m = _SUBJECT.search(claim)
    return m.group(1).strip() if m else ""


def _has_hint(label: str, hints: tuple[str, ...]) -> bool:
    low = label.lower()
    return any(h in low for h in hints)


def _is_subject(label: str, subject: str) -> bool:
    """Метка измерения и субъект claim — про одну и ту же сущность?"""
    low, sub = label.lower().strip(), subject.lower().strip()
    if not sub:
        return False
    return sub in low or low in sub


def _split(measures: list[Measure], subject: str) -> tuple[list[Measure], list[Measure]]:
    """Разделить измерения на «то, о чём claim» и «всё остальное».

    Сначала ищем сущность, названную в самом claim, и только если её в паке
    нет — откатываемся к словарю подсказок. Метка вида «Baseline model»
    попадает в оба словаря сразу, поэтому для роли субъекта признак бейзлайна
    считается сильнее: иначе проверка сравнивала бы сущность саму с собой.
    """
    subject_vals = [m for m in measures if _is_subject(m.label, subject)]
    if not subject_vals:
        subject_vals = [
            m for m in measures
            if _has_hint(m.label, _METHOD_HINTS) and not _has_hint(m.label, _BASELINE_HINTS)
        ]
    rest = [m for m in measures if m not in subject_vals]
    baselines = [m for m in rest if _has_hint(m.label, _BASELINE_HINTS)] or rest
    return subject_vals, baselines


def _wins(a: float, b: float, lower: bool) -> bool:
    """Лучше ли a, чем b, с учётом направления метрики."""
    return a < b - _TOL if lower else a > b + _TOL


def _scale(lower: bool) -> str:
    return "lower-is-better" if lower else "higher-is-better"


def _result(fragment: str, status: str, reason: str, eids: list[str]) -> dict:
    return {"claim_fragment": fragment, "status": status, "reason": reason, "eids": eids}


def _check_direct_number(claim: str, measures: list[Measure], subject: str,
                        series=()) -> dict | None:
    m = _DIRECT.search(claim)
    if not m:
        return None
    claimed = float(m.group(1) or m.group(2))
    fragment = m.group(0)
    # Со строкой таблицы сверяется только утверждение о среднем: «среднее
    # падение для семейства X». Без слова об агрегате имя семейства в claim —
    # это условие эксперимента, а не то, о чём claim, и подстановка среднего
    # сравнивала бы результат модели со средним по всем моделям.
    named_series = (_rank_subject(claim, list(series), subject)
                    if series and _AGGREGATE.search(claim) else [])
    # Как и в ранжировании, догадка о сущности здесь запрещена: сверять число
    # claim'а с числом произвольной строки — это уверенный неверный вердикт.
    subject_vals = named_series or _rank_subject(claim, measures, subject)
    if not subject_vals:
        return _result(
            fragment, STATUS_UNKNOWN,
            "the entity this number belongs to is not named in the evidence pack.", [])
    best = subject_vals[0]
    eids = sorted({x.eid for x in subject_vals})
    if abs(best.value - claimed) < _TOL:
        return _result(
            fragment, STATUS_OK,
            f"claim number {claimed} matches the reported value {best.value}.", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"claim says {claimed}, but the table reports {best.value}.", eids,
    )


def _check_comparison(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _COMPARE.search(claim)
    if not m:
        return None
    target = m.group(1).strip()
    # "outperforms all baselines" — это all/any-утверждение, а не именованное сравнение.
    if re.match(r"(all|any|every|most|the|compared)\b", target, re.I):
        return None
    fragment = m.group(0)
    subject_vals, _ = _split(measures, subject)
    targets = [
        x for x in measures
        if target.lower() in x.label.lower() and x not in subject_vals
    ]
    if not subject_vals or not targets:
        return _result(
            fragment, STATUS_UNKNOWN,
            f"no comparable numbers for '{target}' in the pack.", [],
        )
    m_val, t_val = subject_vals[0].value, targets[0].value
    eids = sorted({subject_vals[0].eid, targets[0].eid})
    lower = _direction([subject_vals[0], targets[0]])
    if lower is _CONFLICT:
        return _result(
            fragment, STATUS_UNKNOWN,
            f"{subject_vals[0].label} and {targets[0].label} are measured on scales that "
            f"run in opposite directions; the numbers are not comparable.", eids,
        )
    if _wins(m_val, t_val, lower):
        return _result(
            fragment, STATUS_OK,
            f"{subject_vals[0].label} ({m_val}) beats {targets[0].label} ({t_val}) "
            f"on a {_scale(lower)} metric.", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"{targets[0].label} ({t_val}) is not worse than {subject_vals[0].label} ({m_val}) "
        f"on a {_scale(lower)} metric; the comparative claim is not supported by "
        f"the numbers.", eids,
    )


def _check_all_any(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _ALL_ANY.search(claim)
    if not m:
        return None
    quantifier = m.group(1).lower()
    fragment = m.group(0)
    subject_vals, baselines = _split(measures, subject)
    if not subject_vals or not baselines:
        return _result(
            fragment, STATUS_UNKNOWN, "no comparable method/baseline numbers in the pack.", []
        )
    m_val = subject_vals[0].value
    label = subject_vals[0].label
    eids = sorted({x.eid for x in subject_vals + baselines})
    lower = _direction(subject_vals[:1] + baselines)
    if lower is _CONFLICT:
        return _result(
            fragment, STATUS_UNKNOWN,
            "the baselines are measured on scales that run in opposite directions; "
            "a single quantified comparison is not defined over them.", eids,
        )
    scale = _scale(lower)
    if quantifier in ("all", "every"):
        offenders = [b for b in baselines if not _wins(m_val, b.value, lower)]
        if offenders:
            b = offenders[0]
            return _result(
                fragment, STATUS_FAILED,
                f"{b.label} ({b.value}) is not worse than {label} ({m_val}) on a {scale} "
                f"metric; the claim that {label} outperforms all baselines "
                f"is not supported by the numbers.", eids,
            )
        return _result(fragment, STATUS_OK,
                       f"every baseline is worse than the method on a {scale} metric.", eids)
    beaten = [b for b in baselines if _wins(m_val, b.value, lower)]
    if not beaten:
        return _result(fragment, STATUS_FAILED,
                       f"the method beats no baseline on a {scale} metric.", eids)
    return _result(fragment, STATUS_OK,
                   f"the method beats at least one baseline on a {scale} metric.", eids)


def _rank_subject(claim: str, measures: list[Measure], subject: str) -> list[Measure]:
    """Измерения, о которых ранжирующий claim, — только по имени, без догадок.

    Для сравнения «A обгоняет B» словарь подсказок безопасен: если сущность не
    нашлась, проверка всё равно сопоставляет метод с бейзлайном. Для claim о
    месте в таблице он опасен: взяв первое попавшееся измерение из семи
    равноправных, чекер выдаёт уверенное failed про сущность, о которой claim
    вообще не говорил. Поэтому здесь молчание лучше догадки.
    """
    named = [m for m in measures if _is_subject(m.label, subject)]
    if named:
        return named
    # Границы слова обязательны: «Model B» находится подстрокой в «model by»,
    # и без них claim «surpassing the next-best model by 37.2 points» назначал
    # субъектом случайную модель из таблицы.
    low = claim.lower()
    return [m for m in measures
            if m.label and re.search(rf"\b{re.escape(m.label.lower())}\b", low)]


def _check_best_worst(claim: str, measures: list[Measure], subject: str,
                      series=()) -> dict | None:
    m = _BEST.search(claim)
    literal = None if m else _BEST_LITERAL.search(claim)
    if not m and not literal:
        return None

    if literal:
        # Буквальное «наибольшее» / «наименьшее»: направление задано самим
        # словом, а не метрикой.
        fragment = literal.group(0)
        lower = literal.group(1).lower() in ("lowest", "smallest")
    else:
        fragment = m.group(0)

    ordinal = _ORDINAL.search(claim)
    if ordinal:
        word = ordinal.group(1).lower()
        claimed_rank = _ORDINAL_WORDS.get(word) or int(ordinal.group(2))
        fragment = f"{ordinal.group(1)} {fragment}"
    else:
        claimed_rank = 1

    # Ряд важнее отдельного числа: claim «семейство X даёт второе по величине
    # среднее падение» говорит о строке таблицы, а не об одной её ячейке.
    # Ранжировать ряды против отдельных чисел нельзя — это разные величины.
    pool = list(series)
    subject_vals = _rank_subject(claim, pool, subject) if pool else []
    if not subject_vals:
        pool = measures
        subject_vals = _rank_subject(claim, pool, subject)
    if not subject_vals:
        return _result(
            fragment, STATUS_UNKNOWN,
            "the entity this ranking claim is about is not named in the evidence pack; "
            "a place in the table cannot be checked without it.", [],
        )

    m_val = subject_vals[0].value
    rivals = [x for x in pool if x is not subject_vals[0]]
    if not rivals:
        return _result(fragment, STATUS_UNKNOWN,
                       "the pack holds a single number; there is nothing to rank it against.",
                       [subject_vals[0].eid])

    if not literal:
        lower = _direction(subject_vals[:1] + rivals)
        if lower is _CONFLICT:
            return _result(
                fragment, STATUS_UNKNOWN,
                "the pack mixes lower-is-better and higher-is-better metrics; "
                "a single ranking over them is not defined.",
                sorted({x.eid for x in pool}),
            )

    better = [x for x in rivals if _wins(x.value, m_val, lower)]
    actual_rank = len(better) + 1
    eids = sorted({x.eid for x in subject_vals[:1] + (better or rivals)})

    if actual_rank == claimed_rank:
        if claimed_rank == 1:
            reason = f"no value in the pack beats {m_val} on a {_scale(lower)} metric."
        else:
            reason = (f"{m_val} ranks {actual_rank} of {len(pool)} on a "
                      f"{_scale(lower)} metric, as claimed.")
        return _result(fragment, STATUS_OK, reason, eids)

    if claimed_rank == 1:
        h = better[0]
        return _result(
            fragment, STATUS_FAILED,
            f"{h.label} ({h.value}) beats the method's {m_val} on a {_scale(lower)} "
            f"metric; the claim that it is {fragment} is contradicted by the numbers.", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"{m_val} ranks {actual_rank} of {len(pool)} on a {_scale(lower)} metric, "
        f"not {claimed_rank} as claimed.", eids,
    )


def _check_mean(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _MEAN.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    if not measures:
        return _result(fragment, STATUS_UNKNOWN, "no comparable numbers in the pack.", [])
    vals = [x.value for x in measures]
    mean = round(sum(vals) / len(vals), 6)
    eids = sorted({x.eid for x in measures})
    if abs(mean - claimed) < _TOL:
        return _result(fragment, STATUS_OK, f"mean of the pack numbers is {mean} as claimed.", eids)
    return _result(
        fragment, STATUS_FAILED, f"mean of the pack numbers is {mean}, not {claimed}.", eids,
    )


def _check_absolute_gain(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _ABS_GAIN.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    subject_vals, baselines = _split(measures, subject)
    if not subject_vals or not baselines:
        return _result(fragment, STATUS_UNKNOWN, "no method/baseline pair to measure a gain.", [])
    m_val, b_val = subject_vals[0].value, baselines[0].value
    eids = sorted({subject_vals[0].eid, baselines[0].eid})
    lower = _direction([subject_vals[0], baselines[0]])
    if lower is _CONFLICT:
        return _result(
            fragment, STATUS_UNKNOWN,
            "the method and the baseline are measured on scales that run in opposite "
            "directions; a gain between them is not defined.", eids,
        )
    # Улучшение на метрике ошибки — это уменьшение, поэтому вычитаемое меняется
    # местами. Иначе честное «снижает RMSE на 12 пунктов» получает знак минус и
    # разъезжается с заявленным приростом.
    delta = round(b_val - m_val if lower else m_val - b_val, 6)
    if abs(delta - claimed) < _TOL:
        return _result(
            fragment, STATUS_OK,
            f"absolute gain {delta} points matches the claim on a {_scale(lower)} metric.",
            eids,
        )
    expression = f"{b_val} - {m_val}" if lower else f"{m_val} - {b_val}"
    return _result(
        fragment, STATUS_FAILED,
        f"absolute gain is {delta} points ({expression}), not the claimed {claimed}.", eids,
    )


def _check_percent_gain(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _PCT_GAIN.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    subject_vals, baselines = _split(measures, subject)
    if not subject_vals or not baselines:
        return _result(fragment, STATUS_UNKNOWN, "no method/baseline pair to measure a gain.", [])
    m_val, b_val = subject_vals[0].value, baselines[0].value
    eids = sorted({subject_vals[0].eid, baselines[0].eid})
    # Относительный прирост от нуля не определён. Это «не с чем сравнивать»,
    # а не противоречие: молча делить нельзя, но и обвинять claim не за что.
    if abs(b_val) < _TOL:
        return _result(
            fragment, STATUS_UNKNOWN,
            f"the baseline value is {b_val}; a relative gain is undefined against a zero base.",
            eids,
        )
    actual = abs(m_val - b_val) / abs(b_val) * 100.0
    if abs(actual - claimed) < _TOL:
        return _result(fragment, STATUS_OK, f"relative gain is {actual:.2f}% as claimed.", eids)
    return _result(
        fragment, STATUS_FAILED,
        f"relative gain is {actual:.2f}% ({round(m_val - b_val, 6)} points over a base of "
        f"{b_val}), not the claimed {claimed}%; note that percentage points and "
        f"percent are different units.", eids,
    )


def check_claim_numbers(claim: str, evidence_pack: list[dict]) -> list[dict]:
    """Прогнать все численные проверки для одного claim по одному evidence pack.

    Возвращает список записей ``{"claim_fragment", "status", "reason", "eids"}``;
    список пуст, если ни одна проверка к claim не подошла.
    """
    measures = _measures(evidence_pack)
    series = _series(evidence_pack)
    subject = _subject(claim)
    checks = (
        _check_direct_number(claim, measures, subject, series),
        _check_comparison(claim, measures, subject),
        _check_all_any(claim, measures, subject),
        _check_best_worst(claim, measures, subject, series),
        _check_mean(claim, measures, subject),
        _check_absolute_gain(claim, measures, subject),
        _check_percent_gain(claim, measures, subject),
    )
    return [r for r in checks if r is not None]
