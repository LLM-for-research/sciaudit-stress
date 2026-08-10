#!/usr/bin/env python3
"""Recursive forbidden-key scan over student-visible files (staff manual §10.1).

``validate_inputs`` checks one Track A input file against one schema. This is
the wider gate: it walks a directory tree, parses every structured file it
understands, and fails if private material — gold labels, stress metadata,
provenance, split assignment — appears anywhere a student could read it. It is
a Required item on the launch checklist (§19.5).

**Roles, not one rule for every file.** A public ``gold.jsonl`` legitimately
carries ``verdict`` and ``supporting_eids`` (Listing 5); a prediction file
legitimately carries the system's own ``verdict`` (Listing 3). Scanning both
with the input rule would fail on correct data, and a gate that cries wolf gets
switched off. So each file is assigned a *profile* from its path, and the
profile decides which subset of ``FORBIDDEN_INPUT_KEYS`` applies:

===========  ====================================================================
input        everything is forbidden (the default, and the fail-closed fallback)
gold         the label itself is expected; stress/provenance/split are not
prediction   the system's own ``verdict`` is expected; nothing else is
internal     not student-visible at all — skipped, with the reason printed
===========  ====================================================================

Every exemption is a line in ``DEFAULT_RULES`` below with a written reason, so
loosening the gate requires a reviewed diff rather than a quiet flag.

Run::

    python -m sciaudit.leakage.forbidden_key_scan data_public/ examples/
    python -m sciaudit.leakage.forbidden_key_scan data_public/ --profile input

Exit codes: 0 clean, 1 leak found, 2 usage/parse error.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

from sciaudit.schemas import FORBIDDEN_INPUT_KEYS, find_forbidden

# Fields a *public gold* file carries by design (staff manual, Listing 5).
GOLD_ALLOWED = frozenset({"gold", "gold_verdict", "verdict", "supporting_eids"})
# A prediction is a system output; its verdict is the thing being measured.
PREDICTION_ALLOWED = frozenset({"verdict"})

#: profile name -> forbidden set (``None`` means "not student-visible, skip")
PROFILES: dict[str, frozenset[str] | None] = {
    "input": FORBIDDEN_INPUT_KEYS,
    "gold": FORBIDDEN_INPUT_KEYS - GOLD_ALLOWED,
    "prediction": FORBIDDEN_INPUT_KEYS - PREDICTION_ALLOWED,
    "internal": None,
}

DEFAULT_PROFILE = "input"

#: (glob, profile, reason). First match wins. A glob containing "/" is matched
#: against the path relative to the scan root; otherwise against the file name.
DEFAULT_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "*internal_annotation*",
        "internal",
        "private-schema example; illustrates the annotation record, never shipped to students",
    ),
    (
        "toy_stress_cases.jsonl",
        "internal",
        "stress methodology teaching material (docs/stress_templates.md); the public half "
        "of each case is what reaches data_public/",
    ),
    ("*gold*.json", "gold", "public gold file (Listing 5)"),
    ("*gold*.jsonl", "gold", "public gold file (Listing 5)"),
    ("*predictions*.jsonl", "prediction", "system output, not a student-facing input (Listing 3)"),
    ("*preds*.jsonl", "prediction", "system output, not a student-facing input (Listing 3)"),
)

STRUCTURED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".yaml", ".yml"}


class ScanError(Exception):
    """A file could not be parsed — treated as a hard failure, not a skip."""


# --- file -> profile ---------------------------------------------------------

def _matches(pattern: str, rel_posix: str, name: str) -> bool:
    target = rel_posix if "/" in pattern else name
    return fnmatch.fnmatch(target, pattern)


def classify(path: Path, root: Path, rules=DEFAULT_RULES) -> tuple[str, str]:
    """Return ``(profile, reason)`` for one file."""
    try:
        rel_posix = path.relative_to(root).as_posix()
    except ValueError:
        rel_posix = path.as_posix()
    for pattern, profile, reason in rules:
        if _matches(pattern, rel_posix, path.name):
            return profile, reason
    return DEFAULT_PROFILE, "default: treated as student-visible input"


def load_rules(path: Path) -> tuple[tuple[str, str, str], ...]:
    """Load a rules override file: ``[{"glob":…, "profile":…, "reason":…}, …]``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ScanError(f"{path}: cannot read rules file: {e}") from e
    rules = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "glob" not in item or "profile" not in item:
            raise ScanError(f"{path}: rule {i} needs at least 'glob' and 'profile'")
        if item["profile"] not in PROFILES:
            raise ScanError(f"{path}: rule {i} has unknown profile {item['profile']!r}")
        rules.append((item["glob"], item["profile"], item.get("reason", "(no reason given)")))
    return tuple(rules)


# --- parsing -----------------------------------------------------------------

def parse_objects(path: Path) -> list[tuple[str, object]]:
    """Return ``[(location_label, parsed_object), …]`` for a structured file."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")

    if suffix in {".jsonl", ".ndjson"}:
        objects = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                objects.append((f":{line_no}", json.loads(line)))
            except json.JSONDecodeError as e:
                raise ScanError(f"{path}:{line_no}: invalid JSON: {e}") from e
        return objects

    if suffix == ".json":
        try:
            return [("", json.loads(text))]
        except json.JSONDecodeError as e:
            raise ScanError(f"{path}: invalid JSON: {e}") from e

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            return []  # optional format; absence of PyYAML must not fake a pass
        try:
            return [("", doc) for doc in yaml.safe_load_all(text) if doc is not None]
        except yaml.YAMLError as e:
            raise ScanError(f"{path}: invalid YAML: {e}") from e

    return []


def iter_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.is_file())


# --- scanning ----------------------------------------------------------------

def scan_file(
    path: Path,
    root: Path,
    rules=DEFAULT_RULES,
    forced_profile: str | None = None,
) -> tuple[str, str, list[str]]:
    """Scan one file. Returns ``(profile, reason, leaks)``."""
    if forced_profile:
        profile, reason = forced_profile, "forced via --profile"
    else:
        profile, reason = classify(path, root, rules)

    forbidden = PROFILES[profile]
    if forbidden is None:
        return profile, reason, []

    leaks = []
    for label, obj in parse_objects(path):
        for hit in find_forbidden(obj, forbidden):
            leaks.append(f"{path}{label}: [{profile}] {hit}")
    return profile, reason, leaks


def scan(
    targets: list[Path],
    rules=DEFAULT_RULES,
    forced_profile: str | None = None,
    verbose: bool = False,
) -> tuple[list[str], int]:
    """Scan files/directories. Returns ``(leaks, files_examined)``."""
    leaks: list[str] = []
    examined = 0

    for target in targets:
        if not target.exists():
            raise ScanError(f"{target}: no such file or directory")
        root = target if target.is_dir() else target.parent

        for path in iter_files(target):
            if path.suffix.lower() not in STRUCTURED_SUFFIXES:
                continue
            examined += 1
            profile, reason, file_leaks = scan_file(path, root, rules, forced_profile)
            leaks.extend(file_leaks)
            if verbose:
                if PROFILES[profile] is None:
                    print(f"  SKIP  {path} [{profile}] — {reason}")
                else:
                    status = "LEAK" if file_leaks else "ok"
                    print(f"  {status:5} {path} [{profile}]")

    return leaks, examined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan student-visible files for leaked private fields (§10.1).",
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to scan.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="Force one profile for every file instead of deriving it from the path.",
    )
    parser.add_argument("--rules", type=Path, help="JSON file overriding the built-in rules.")
    parser.add_argument("-v", "--verbose", action="store_true", help="List every file examined.")
    args = parser.parse_args(argv)

    try:
        rules = load_rules(args.rules) if args.rules else DEFAULT_RULES
        leaks, examined = scan(
            [Path(p) for p in args.paths], rules, args.profile, args.verbose
        )
    except ScanError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if leaks:
        for leak in leaks:
            print(f"LEAK: {leak}")
        print(
            f"forbidden_key_scan: FAIL — {len(leaks)} leaked field(s) "
            f"across {examined} file(s)"
        )
        return 1

    print(f"forbidden_key_scan: OK — {examined} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
