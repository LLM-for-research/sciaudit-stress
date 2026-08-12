#!/usr/bin/env python3
"""Deterministic numeric/table checker for Track A (staff manual §11.5).

A dependency-free, offline component that checks the numbers in a claim
against the numbers in an evidence pack. It never calls a model — it is a
building block for the B3 baseline and for human-verification tooling.

Implemented checks (of the §11.5 catalogue):

1. direct numeric claim vs evidence ("achieves 91.0" vs a reported 88.1);
2. comparative claim ("Method X outperforms Baseline B");
3. all/any claim ("outperforms all baselines") vs every/any baseline;
4. best/worst and rank claims ("highest", "best", "ranks first");
5. mean/average claims vs the mean of the pack numbers;
6. absolute gain m - b (points / percentage points);
7. relative gain (m - b) / b in percent — including the
   percent-vs-percentage-point distinction (a gain of 10 percentage points
   is not the same as a 10% gain, a typical delegation-log failure).

Each check yields a record::

    {"claim_fragment", "status", "reason", "eids"}

where ``status`` is "ok" (numbers agree), "failed" (a numerical contradiction
was found) or "unknown" (nothing comparable in the pack).

Reference case (§11.5): claim "Method X outperforms all baselines", evidence
"Method X = 91.2, Baseline C = 92.1" -> status "failed" with the eid of the
table row.
"""
from __future__ import annotations

import re

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"

_NUM = re.compile(r"\d+(?:\.\d+)?")
# "Method X = 91.2" / "Method Y: 93.4" style measurements.
_PAIR = re.compile(r"([A-Za-z][A-Za-z0-9 .\-']{0,40}?)\s*[=:]\s*(\d+(?:\.\d+)?)")

_METHOD_HINTS = (
    "method", "ours", "proposed", "our model", "full model",
    "our approach", "model",
)
_BASELINE_HINTS = ("baseline", "sota", "previous", "competitor", "without", "standard")

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


class Measure:
    """One number extracted from an evidence unit."""

    __slots__ = ("label", "value", "eid")

    def __init__(self, label: str, value: float, eid: str):
        self.label = label
        self.value = value
        self.eid = eid


def _measures(evidence_pack: list[dict]) -> list[Measure]:
    """Parse "Label = value" style numbers out of the evidence pack."""
    out: list[Measure] = []
    for unit in evidence_pack or []:
        if not isinstance(unit, dict):
            continue
        eid = str(unit.get("eid", "?"))
        text = str(unit.get("text", ""))
        for label, value in _PAIR.findall(text):
            out.append(Measure(label.strip(), float(value), eid))
        for nb in unit.get("normalized_numbers", []) or []:
            if isinstance(nb, dict) and isinstance(nb.get("value"), (int, float)):
                out.append(Measure(str(nb.get("context", "")), float(nb["value"]), eid))
    return out


def _method_vals(measures: list[Measure]) -> list[Measure]:
    return [m for m in measures if any(h in m.label.lower() for h in _METHOD_HINTS)]


def _baseline_vals(measures: list[Measure]) -> list[Measure]:
    return [m for m in measures if any(h in m.label.lower() for h in _BASELINE_HINTS)]


def _result(fragment: str, status: str, reason: str, eids: list[str]) -> dict:
    return {"claim_fragment": fragment, "status": status, "reason": reason, "eids": eids}


def _check_direct_number(claim: str, measures: list[Measure]) -> dict | None:
    m = _DIRECT.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    methods = _method_vals(measures)
    if not methods:
        return _result(fragment, STATUS_UNKNOWN, "no comparable numbers in the pack.", [])
    best = methods[0]
    eids = sorted({x.eid for x in methods})
    if abs(best.value - claimed) < 1e-9:
        return _result(
            fragment, STATUS_OK,
            f"claim number {claimed} matches the reported value {best.value}.", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"claim says {claimed}, but the table reports {best.value}.", eids,
    )


def _check_comparison(claim: str, measures: list[Measure]) -> dict | None:
    m = _COMPARE.search(claim)
    if not m:
        return None
    target = m.group(1).strip()
    # "outperforms all baselines" is an all/any claim, not a named comparison.
    if re.match(r"(all|any|every|most|the|compared)\b", target, re.I):
        return None
    fragment = m.group(0)
    methods = _method_vals(measures)
    targets = [x for x in measures if target.lower() in x.label.lower()]
    if not methods or not targets:
        return _result(
            fragment, STATUS_UNKNOWN,
            f"no comparable numbers for '{target}' in the pack.", [],
        )
    m_val, t_val = methods[0].value, targets[0].value
    eids = sorted({methods[0].eid, targets[0].eid})
    if m_val > t_val + 1e-9:
        return _result(
            fragment, STATUS_OK,
            f"{methods[0].label} ({m_val}) exceeds {targets[0].label} ({t_val}).", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"{targets[0].label} ({t_val}) is not below {methods[0].label} ({m_val}); "
        f"the comparative claim is not supported by the numbers.", eids,
    )


def _check_all_any(claim: str, measures: list[Measure]) -> dict | None:
    m = _ALL_ANY.search(claim)
    if not m:
        return None
    quantifier = m.group(1).lower()
    fragment = m.group(0)
    methods = _method_vals(measures)
    baselines = _baseline_vals(measures)
    if not methods or not baselines:
        return _result(
            fragment, STATUS_UNKNOWN, "no comparable method/baseline numbers in the pack.", []
        )
    m_val = methods[0].value
    eids = sorted({x.eid for x in methods + baselines})
    if quantifier == "all" or quantifier == "every":
        offenders = [b for b in baselines if b.value >= m_val - 1e-9]
        if offenders:
            b = offenders[0]
            return _result(
                fragment, STATUS_FAILED,
                f"{b.label} ({b.value}) is not below {methods[0].label} ({m_val}); "
                f"the claim that {methods[0].label} outperforms all baselines "
                f"is not supported by the numbers.", eids,
            )
        return _result(fragment, STATUS_OK, "every baseline is below the method.", eids)
    better = [b for b in baselines if b.value < m_val - 1e-9]
    if not better:
        return _result(fragment, STATUS_FAILED, "no baseline is below the method.", eids)
    return _result(fragment, STATUS_OK, "at least one baseline is below the method.", eids)


def _check_best_worst(claim: str, measures: list[Measure]) -> dict | None:
    m = _BEST.search(claim)
    if not m:
        return None
    fragment = m.group(0)
    methods = _method_vals(measures)
    if not methods:
        return _result(fragment, STATUS_UNKNOWN, "no comparable numbers in the pack.", [])
    m_val = methods[0].value
    eids = sorted({x.eid for x in measures if x.value >= m_val - 1e-9})
    higher = [x for x in measures if x.value > m_val + 1e-9]
    if higher:
        h = higher[0]
        return _result(
            fragment, STATUS_FAILED,
            f"{h.label} ({h.value}) exceeds the method's {m_val}; the claim that it "
            f"is {fragment} is contradicted by the numbers.", eids,
        )
    return _result(fragment, STATUS_OK, f"no value in the pack exceeds {m_val}.", eids)


def _check_mean(claim: str, measures: list[Measure]) -> dict | None:
    m = _MEAN.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    if not measures:
        return _result(fragment, STATUS_UNKNOWN, "no comparable numbers in the pack.", [])
    vals = [x.value for x in measures]
    mean = sum(vals) / len(vals)
    eids = sorted({x.eid for x in measures})
    if abs(mean - claimed) < 1e-9:
        return _result(fragment, STATUS_OK, f"mean of the pack numbers is {mean} as claimed.", eids)
    return _result(
        fragment, STATUS_FAILED, f"mean of the pack numbers is {mean}, not {claimed}.", eids,
    )


def _check_absolute_gain(claim: str, measures: list[Measure]) -> dict | None:
    m = _ABS_GAIN.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    methods, baselines = _method_vals(measures), _baseline_vals(measures)
    if not methods or not baselines:
        return _result(fragment, STATUS_UNKNOWN, "no method/baseline pair to measure a gain.", [])
    m_val, b_val = methods[0].value, baselines[0].value
    delta = m_val - b_val
    eids = sorted({methods[0].eid, baselines[0].eid})
    if abs(delta - claimed) < 1e-9:
        return _result(
            fragment, STATUS_OK,
            f"absolute gain {delta} points matches the claim.", eids,
        )
    return _result(
        fragment, STATUS_FAILED,
        f"absolute gain is {delta} points ({m_val} - {b_val}), not the claimed {claimed}.", eids,
    )


def _check_percent_gain(claim: str, measures: list[Measure]) -> dict | None:
    m = _PCT_GAIN.search(claim)
    if not m:
        return None
    claimed = float(m.group(1))
    fragment = m.group(0)
    methods, baselines = _method_vals(measures), _baseline_vals(measures)
    if not methods or not baselines:
        return _result(fragment, STATUS_UNKNOWN, "no method/baseline pair to measure a gain.", [])
    m_val, b_val = methods[0].value, baselines[0].value
    actual = abs(m_val - b_val) / b_val * 100.0
    eids = sorted({methods[0].eid, baselines[0].eid})
    if abs(actual - claimed) < 1e-9:
        return _result(fragment, STATUS_OK, f"relative gain is {actual:.2f}% as claimed.", eids)
    return _result(
        fragment, STATUS_FAILED,
        f"relative gain is {actual:.2f}% ({(m_val - b_val):.2f} points over a base of "
        f"{b_val}), not the claimed {claimed}%; note that percentage points and "
        f"percent are different units.", eids,
    )


def check_claim_numbers(claim: str, evidence_pack: list[dict]) -> list[dict]:
    """Run every numeric check for one claim against one evidence pack.

    Returns a list of ``{"claim_fragment", "status", "reason", "eids"}``
    records; the list is empty when no check matched the claim.
    """
    measures = _measures(evidence_pack)
    checks = (
        _check_direct_number(claim, measures),
        _check_comparison(claim, measures),
        _check_all_any(claim, measures),
        _check_best_worst(claim, measures),
        _check_mean(claim, measures),
        _check_absolute_gain(claim, measures),
        _check_percent_gain(claim, measures),
    )
    return [r for r in checks if r is not None]
