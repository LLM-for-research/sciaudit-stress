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

# Keys that only ever exist in private/internal data. Their presence anywhere
# in a public input object means gold labels, stress metadata, provenance, or
# private rationale leaked into the student-facing layer.
FORBIDDEN_INPUT_KEYS = frozenset({
    "gold",
    "gold_verdict",
    "verdict",
    "supporting_eids",
    "severity",
    "stress",
    "stress_type",
    "is_stress_case",
    "seed_instance_id",
    "private_rationale",
    "rationale_private",
    "review",
    "review_note",
    "human_verification_note",
    "validation_level",
    "reviewer",
    "provenance",
    "provenance_map",
    "provenance_ref",
    "source_url",
    "source_ref",
    "source_location",
    "is_distractor",
    "split",
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


def find_forbidden_keys(obj, forbidden=FORBIDDEN_INPUT_KEYS, path="$") -> list[str]:
    """Recursively locate forbidden keys; returns JSONPath-like locations."""
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}"
            if key in forbidden:
                hits.append(key_path)
            hits.extend(find_forbidden_keys(value, forbidden, key_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(find_forbidden_keys(item, forbidden, f"{path}[{i}]"))
    return hits


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
