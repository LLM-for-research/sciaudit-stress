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
"""
from __future__ import annotations

import re

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"

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

_DIRECT = re.compile(
    r"(?:achieves|reports|reaches|attains|accuracy of|score of|latency of|"
    r"reduces\s+\w+\s+to|improves\s+\w+\s+to)\s+(\d+(?:\.\d+)?)",
    re.I,
)
_COMPARE = re.compile(r"outperform\w*\s+(?:the\s+)?([A-Z][A-Za-z0-9 \-']*)", re.I)
_ALL_ANY = re.compile(
    r"outperform\w*\s+(all|every|any)\s+(?:the\s+)?(?:compared\s+)?baselines?", re.I
)
_BEST = re.compile(r"(highest|best|largest|ranks?\s+first|top[-\s]?1)", re.I)
_MEAN = re.compile(r"(?:mean|average)\s+(?:accuracy|score|latency|performance)?\s*"
                   r"(?:of\s*)?(?:is\s+)?(\d+(?:\.\d+)?)", re.I)
_ABS_GAIN = re.compile(
    r"(?:improve\w*|increase\w*|gain\w*)\s+.{0,60}?\bby\s+(\d+(?:\.\d+)?)\s+"
    r"(?:percentage\s+)?points?", re.I,
)
_PCT_GAIN = re.compile(
    r"(?:improve\w*|increase\w*|reduc\w*)\s+.{0,60}?\bby\s+(\d+(?:\.\d+)?)\s*%", re.I
)

_TOL = 1e-9


class Measure:
    """Одно число, извлечённое из единицы evidence."""

    __slots__ = ("label", "value", "eid")

    def __init__(self, label: str, value: float, eid: str):
        self.label = label
        self.value = value
        self.eid = eid


def _measures(evidence_pack: list[dict]) -> list[Measure]:
    """Разобрать числа из evidence pack: пары «метка = значение» и «A n vs B m»."""
    out: list[Measure] = []
    seen: set[tuple[str, float, str]] = set()

    def add(label: str, value: float, eid: str) -> None:
        label = label.strip()
        key = (label.lower(), value, eid)
        if key not in seen:
            seen.add(key)
            out.append(Measure(label, value, eid))

    for unit in evidence_pack or []:
        if not isinstance(unit, dict):
            continue
        eid = str(unit.get("eid", "?"))
        text = str(unit.get("text", ""))
        for label, value in _PAIR.findall(text):
            add(label, float(value), eid)
        for left_label, left_value, right_label, right_value in _VS_PAIR.findall(text):
            add(left_label, float(left_value), eid)
            add(right_label, float(right_value), eid)
        for nb in unit.get("normalized_numbers", []) or []:
            if isinstance(nb, dict) and isinstance(nb.get("value"), (int, float)):
                add(str(nb.get("context", "")), float(nb["value"]), eid)
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


def _result(fragment: str, status: str, reason: str, eids: list[str]) -> dict:
    return {"claim_fragment": fragment, "status": status, "reason": reason, "eids": eids}


def _check_direct_number(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _DIRECT.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    subject_vals, _ = _split(measures, subject)
    if not subject_vals:
        return _result(fragment, STATUS_UNKNOWN, "no comparable numbers in the pack.", [])
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
    if m_val > t_val + _TOL:
        return _result(
            fragment, STATUS_OK,
            f"{subject_vals[0].label} ({m_val}) exceeds {targets[0].label} ({t_val}).", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"{targets[0].label} ({t_val}) is not below {subject_vals[0].label} ({m_val}); "
        f"the comparative claim is not supported by the numbers.", eids,
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
    if quantifier in ("all", "every"):
        offenders = [b for b in baselines if b.value >= m_val - _TOL]
        if offenders:
            b = offenders[0]
            return _result(
                fragment, STATUS_FAILED,
                f"{b.label} ({b.value}) is not below {label} ({m_val}); "
                f"the claim that {label} outperforms all baselines "
                f"is not supported by the numbers.", eids,
            )
        return _result(fragment, STATUS_OK, "every baseline is below the method.", eids)
    better = [b for b in baselines if b.value < m_val - _TOL]
    if not better:
        return _result(fragment, STATUS_FAILED, "no baseline is below the method.", eids)
    return _result(fragment, STATUS_OK, "at least one baseline is below the method.", eids)


def _check_best_worst(claim: str, measures: list[Measure], subject: str) -> dict | None:
    m = _BEST.search(claim)
    if not m:
        return None
    fragment = m.group(0)
    subject_vals, _ = _split(measures, subject)
    if not subject_vals:
        return _result(fragment, STATUS_UNKNOWN, "no comparable numbers in the pack.", [])
    m_val = subject_vals[0].value
    eids = sorted({x.eid for x in measures if x.value >= m_val - _TOL})
    higher = [x for x in measures if x.value > m_val + _TOL]
    if higher:
        h = higher[0]
        return _result(
            fragment, STATUS_FAILED,
            f"{h.label} ({h.value}) exceeds the method's {m_val}; the claim that it "
            f"is {fragment} is contradicted by the numbers.", eids,
        )
    return _result(fragment, STATUS_OK, f"no value in the pack exceeds {m_val}.", eids)


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
    delta = round(m_val - b_val, 6)
    eids = sorted({subject_vals[0].eid, baselines[0].eid})
    if abs(delta - claimed) < _TOL:
        return _result(
            fragment, STATUS_OK,
            f"absolute gain {delta} points matches the claim.", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"absolute gain is {delta} points ({m_val} - {b_val}), not the claimed {claimed}.", eids,
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
    subject = _subject(claim)
    checks = (
        _check_direct_number(claim, measures, subject),
        _check_comparison(claim, measures, subject),
        _check_all_any(claim, measures, subject),
        _check_best_worst(claim, measures, subject),
        _check_mean(claim, measures, subject),
        _check_absolute_gain(claim, measures, subject),
        _check_percent_gain(claim, measures, subject),
    )
    return [r for r in checks if r is not None]
