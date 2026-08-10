#!/usr/bin/env python3
"""Validate Track A public input JSONL files.

Checks, per line:
  1. JSON Schema validity against ``schemas/track_a_input.schema.json``;
  2. no private fields (gold verdict, severity, stress metadata, private
     rationale, provenance, review fields, split) anywhere in the object;
  3. unique ``instance_id`` across the file and unique ``eid`` within a pack;
  4. ``allowed_evidence_ids`` is a subset of the evidence-pack eids.

Run:
    python -m sciaudit.schemas.validate_inputs <inputs.jsonl> [...]
Exit code 0 if every file is valid, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys

from . import (
    FORBIDDEN_INPUT_KEYS,
    find_forbidden,
    load_schema,
    read_jsonl,
    report,
    schema_errors,
)

SCHEMA_NAME = "track_a_input.schema.json"


def validate_input_file(path: str) -> list[str]:
    """Return a list of error strings for one JSONL file (empty if valid)."""
    schema = load_schema(SCHEMA_NAME)
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError) as e:
        return [f"{path}: {e}"]

    problems: list[str] = []
    seen_ids: set[str] = set()

    for line_no, obj in rows:
        where = f"{path}:{line_no}"

        leaks = find_forbidden(obj, FORBIDDEN_INPUT_KEYS)
        for loc in leaks:
            problems.append(
                f"{where}: private field leaked into public input: {loc}"
            )

        for err in schema_errors(obj, schema):
            problems.append(f"{where}: schema: {err}")

        instance_id = obj.get("instance_id")
        if isinstance(instance_id, str):
            if instance_id in seen_ids:
                problems.append(f"{where}: duplicate instance_id: {instance_id}")
            seen_ids.add(instance_id)

        pack = obj.get("evidence_pack")
        if isinstance(pack, list):
            eids = [e.get("eid") for e in pack if isinstance(e, dict)]
            dupes = {e for e in eids if e is not None and eids.count(e) > 1}
            for e in sorted(dupes):
                problems.append(f"{where}: duplicate eid in evidence_pack: {e}")

            allowed = obj.get("allowed_evidence_ids")
            if isinstance(allowed, list):
                unknown = [a for a in allowed if a not in eids]
                for a in unknown:
                    problems.append(
                        f"{where}: allowed_evidence_ids contains unknown eid: {a}"
                    )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Track A public input JSONL.")
    parser.add_argument("paths", nargs="+", help="Input JSONL file(s) to validate.")
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        problems = validate_input_file(path)
        try:
            total = len(read_jsonl(path))
        except (OSError, ValueError):
            total = 0
        exit_code |= report(f"validate_inputs {path}", total, problems)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
