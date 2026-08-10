"""Shared helpers for schema validation CLIs.

JSON Schemas live in the top-level ``schemas/`` directory. These helpers add
the checks JSON Schema alone cannot express: private-field (leakage) scanning,
duplicate instance IDs, evidence-ID consistency, and prediction/input
cross-checks.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "schemas"

FORBIDDEN_INPUT_KEYS = frozenset({
    # --- gold labels ---
    "gold",
    "gold_verdict",
    "verdict",
    "expected_verdict",
    "supporting_eids",
    "severity",
    # --- stress metadata ---
    "stress",
    "stress_type",
    "transformation_type",
    "is_stress_case",
    "seed_instance_id",
    "evidence_removal",
    "scope_expansion",
    "numeric_perturbation",
    "distractor_flag",
    # --- private rationale / review ---
    "private_rationale",
    "rationale_private",
    "review",
    "review_note",
    "human_verification_note",
    "validation_level",
    "reviewer",
    "adjudication_note",
    "TA_note",
    # --- provenance / licensing ---
    "provenance",
    "provenance_map",
    "provenance_ref",
    "source_url",
    "source_ref",
    "source_location",
    "paper_title",
    "authors",
    "venue",
    "license_status",
    "is_distractor",
    # --- split assignment (leaks which slice an instance belongs to) ---
    "split",
    "private_slice",
    "GoldHidden",
    "AutoStressHidden",
    "ChallengeHidden",
})


def load_schema(name: str) -> dict:
    path = SCHEMAS_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[tuple[int, dict]]:
    """Return [(line_number, object), ...]; raises ValueError on bad JSON."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no}: expected a JSON object")
            rows.append((line_no, obj))
    return rows


def find_forbidden(
    obj,
    forbidden=FORBIDDEN_INPUT_KEYS,
    path="$",
    *,
    scan_values: bool = True,
) -> list[str]:
    """Recursively locate forbidden keys *and* forbidden string values.

    Staff manual requires both directions: a private name is a
    leak whether it appears as ``{"stress_type": ...}`` or as
    ``{"slice": "AutoStressHidden"}``. Matching on values is exact equality, so
    prose that merely mentions a forbidden word ("the authors report ...") does
    not trip the scan.

    Returns JSONPath-like locations; value hits are rendered as
    ``$.slice == 'AutoStressHidden'``.
    """
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}"
            if key in forbidden:
                hits.append(key_path)
            elif scan_values and isinstance(value, str) and value in forbidden:
                hits.append(f"{key_path} == {value!r}")
            hits.extend(find_forbidden(value, forbidden, key_path, scan_values=scan_values))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            item_path = f"{path}[{i}]"
            if scan_values and isinstance(item, str) and item in forbidden:
                hits.append(f"{item_path} == {item!r}")
            hits.extend(find_forbidden(item, forbidden, item_path, scan_values=scan_values))
    return hits


def find_forbidden_keys(obj, forbidden=FORBIDDEN_INPUT_KEYS, path="$") -> list[str]:
    """Key-only variant of :func:`find_forbidden` (kept for callers that
    deliberately want to ignore values)."""
    return find_forbidden(obj, forbidden, path, scan_values=False)


def schema_errors(instance: dict, schema: dict) -> list[str]:
    """Validate one object against a JSON Schema; returns error strings."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        loc = "$" + "".join(
            f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path
        )
        errors.append(f"{loc}: {err.message}")
    return errors


def report(name: str, total: int, problems: list[str]) -> int:
    """Print a summary; return process exit code (0 ok, 1 fail)."""
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        print(f"{name}: FAIL — {len(problems)} error(s) across {total} object(s)")
        return 1
    print(f"{name}: OK — {total} valid object(s)")
    return 0
