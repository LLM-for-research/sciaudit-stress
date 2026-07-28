#!/usr/bin/env python3
"""Validate prediction JSONL files.

Checks, per line:
  1. JSON Schema validity against ``schemas/prediction.schema.json``
     (fixed verdict enum, documented issue-tag enum, confidence in [0, 1]);
  2. unique ``instance_id`` across the file.

With ``--input inputs.jsonl`` also cross-checks that:
  3. every prediction targets a known instance and every instance is covered;
  4. ``predicted_eids`` only cite that instance's ``allowed_evidence_ids``.

Run:
    python -m sciaudit.schemas.validate_predictions <predictions.jsonl> [--input inputs.jsonl]
Exit code 0 if valid, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys

from . import load_schema, read_jsonl, report, schema_errors

SCHEMA_NAME = "prediction.schema.json"


def validate_prediction_file(path: str, input_path: str | None = None) -> list[str]:
    """Return a list of error strings for one predictions file (empty if valid)."""
    schema = load_schema(SCHEMA_NAME)
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError) as e:
        return [f"{path}: {e}"]

    allowed_by_instance: dict[str, set[str]] | None = None
    if input_path is not None:
        try:
            input_rows = read_jsonl(input_path)
        except (OSError, ValueError) as e:
            return [f"{input_path}: {e}"]
        allowed_by_instance = {
            obj["instance_id"]: set(obj.get("allowed_evidence_ids", []))
            for _, obj in input_rows
            if isinstance(obj.get("instance_id"), str)
        }

    problems: list[str] = []
    seen_ids: set[str] = set()

    for line_no, obj in rows:
        where = f"{path}:{line_no}"

        for err in schema_errors(obj, schema):
            problems.append(f"{where}: schema: {err}")

        instance_id = obj.get("instance_id")
        if isinstance(instance_id, str):
            if instance_id in seen_ids:
                problems.append(f"{where}: duplicate instance_id: {instance_id}")
            seen_ids.add(instance_id)

        if allowed_by_instance is not None and isinstance(instance_id, str):
            if instance_id not in allowed_by_instance:
                problems.append(f"{where}: prediction for unknown instance_id: {instance_id}")
            else:
                allowed = allowed_by_instance[instance_id]
                eids = obj.get("predicted_eids")
                if isinstance(eids, list):
                    for eid in eids:
                        if eid not in allowed:
                            problems.append(
                                f"{where}: predicted_eids cites eid not in "
                                f"allowed_evidence_ids: {eid}"
                            )

    if allowed_by_instance is not None:
        missing = sorted(set(allowed_by_instance) - seen_ids)
        for instance_id in missing:
            problems.append(f"{path}: missing prediction for instance_id: {instance_id}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate prediction JSONL.")
    parser.add_argument("paths", nargs="+", help="Prediction JSONL file(s) to validate.")
    parser.add_argument(
        "--input",
        default=None,
        help="Optional Track A input JSONL to cross-check coverage and evidence IDs.",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        problems = validate_prediction_file(path, args.input)
        try:
            total = len(read_jsonl(path))
        except (OSError, ValueError):
            total = 0
        exit_code |= report(f"validate_predictions {path}", total, problems)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
