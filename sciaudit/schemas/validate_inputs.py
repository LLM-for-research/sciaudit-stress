#!/usr/bin/env python3
"""Валидация публичных входных JSONL Track A.

Проверяется построчно:
  1. соответствие JSON Schema ``schemas/track_a_input.schema.json``;
  2. отсутствие приватных полей (gold-вердикт, severity, стресс-метаданные,
     приватное обоснование, провенанс, поля ревью, сплит) на любой глубине;
  3. уникальность ``instance_id`` в пределах файла и уникальность ``eid``
     внутри одного пака;
  4. ``allowed_evidence_ids`` — подмножество eid из evidence pack.

Запуск:
    python -m sciaudit.schemas.validate_inputs <inputs.jsonl> [...]
Код возврата 0, если валидны все файлы, иначе 1.
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
    """Вернуть список ошибок для одного JSONL (пустой, если файл валиден)."""
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
                f"{where}: приватное поле просочилось в публичный вход: {loc}"
            )

        for err in schema_errors(obj, schema):
            problems.append(f"{where}: схема: {err}")

        instance_id = obj.get("instance_id")
        if isinstance(instance_id, str):
            if instance_id in seen_ids:
                problems.append(f"{where}: дубликат instance_id: {instance_id}")
            seen_ids.add(instance_id)

        pack = obj.get("evidence_pack")
        if isinstance(pack, list):
            eids = [e.get("eid") for e in pack if isinstance(e, dict)]
            dupes = {e for e in eids if e is not None and eids.count(e) > 1}
            for e in sorted(dupes):
                problems.append(f"{where}: дубликат eid в evidence_pack: {e}")

            allowed = obj.get("allowed_evidence_ids")
            if isinstance(allowed, list):
                unknown = [a for a in allowed if a not in eids]
                for a in unknown:
                    problems.append(
                        f"{where}: allowed_evidence_ids содержит неизвестный eid: {a}"
                    )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Валидация публичных входных JSONL Track A.",
    )
    parser.add_argument("paths", nargs="+", help="Файл(ы) JSONL со входами для проверки.")
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
