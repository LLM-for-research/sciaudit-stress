"""Deterministic numeric/table checker tests (issue #15).

Covers the §11.5 catalogue: direct numeric claims, percent vs
percentage-point, mean, best/worst, rank, all/any, absolute and relative
gains, plus the mandatory reference case, eid reporting, and offline
operation (no model, no network).
"""
import pytest

from sciaudit.baselines.numeric_checker import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNKNOWN,
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


# --- mandatory reference case (staff manual §11.5) ---------------------------

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


# --- direct numeric claims ----------------------------------------------------

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


# --- percent vs percentage-point ----------------------------------------------

def test_percentage_points_match_is_ok():
    results = check_claim_numbers(
        "Method X improves the accuracy by 10 percentage points.",
        _pack("Method X = 88.1, Baseline = 78.1"),
    )
    assert _statuses(results) == {STATUS_OK}


def test_percent_is_not_percentage_points():
    # 10 percentage points over a base of 78.1 is ~12.8%, not 10%.
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


# --- mean ---------------------------------------------------------------------

def test_mean_check():
    pack = _pack("Method X = 89.0", "Method X = 91.0")
    ok = check_claim_numbers("The mean accuracy is 90.0.", pack)
    bad = check_claim_numbers("The mean accuracy is 90.5.", pack)
    assert _statuses(ok) == {STATUS_OK}
    assert _failed(bad) and any("90.0" in r["reason"] for r in _failed(bad))


# --- best / worst / rank -------------------------------------------------------

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


# --- comparative claim ---------------------------------------------------------

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


# --- all/any --------------------------------------------------------------------

def test_all_baselines_beaten_is_ok():
    results = check_claim_numbers(
        "Method X outperforms all baselines.",
        _pack("Method X = 91.2, Baseline B = 90.8, Baseline C = 89.9"),
    )
    assert _statuses(results) == {STATUS_OK}


# --- absolute gain ---------------------------------------------------------------

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


# --- edge cases -------------------------------------------------------------------

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
