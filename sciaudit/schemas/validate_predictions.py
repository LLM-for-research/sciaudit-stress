#!/usr/bin/env python3
"""Валидация JSONL с предсказаниями.

Проверяется построчно:
  1. соответствие JSON Schema ``schemas/prediction.schema.json``
     (фиксированный enum вердиктов, документированный enum issue-тегов,
     confidence в [0, 1]);
  2. уникальность ``instance_id`` в пределах файла.

С флагом ``--input inputs.jsonl`` дополнительно сверяется, что:
  3. каждое предсказание относится к известному инстансу и каждый инстанс покрыт;
  4. ``predicted_eids`` ссылается только на ``allowed_evidence_ids`` своего инстанса.

Запуск:
    python -m sciaudit.schemas.validate_predictions <predictions.jsonl> [--input inputs.jsonl]
Код возврата 0, если всё валидно, иначе 1.
"""
from __future__ import annotations

import argparse
import sys

from . import load_schema, read_jsonl, report, schema_errors

SCHEMA_NAME = "prediction.schema.json"


def validate_prediction_file(path: str, input_path: str | None = None) -> list[str]:
    """Вернуть список ошибок для одного файла предсказаний (пустой, если валиден)."""
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
            problems.append(f"{where}: схема: {err}")

        instance_id = obj.get("instance_id")
        if isinstance(instance_id, str):
            if instance_id in seen_ids:
                problems.append(f"{where}: дубликат instance_id: {instance_id}")
            seen_ids.add(instance_id)

        if allowed_by_instance is not None and isinstance(instance_id, str):
            if instance_id not in allowed_by_instance:
                problems.append(
                    f"{where}: предсказание для неизвестного instance_id: {instance_id}"
                )
            else:
                allowed = allowed_by_instance[instance_id]
                eids = obj.get("predicted_eids")
                if isinstance(eids, list):
                    for eid in eids:
                        if eid not in allowed:
                            problems.append(
                                f"{where}: predicted_eids ссылается на eid вне "
                                f"allowed_evidence_ids: {eid}"
                            )

    if allowed_by_instance is not None:
        missing = sorted(set(allowed_by_instance) - seen_ids)
        for instance_id in missing:
            problems.append(f"{path}: нет предсказания для instance_id: {instance_id}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Валидация JSONL с предсказаниями.")
    parser.add_argument("paths", nargs="+", help="Файл(ы) JSONL с предсказаниями.")
    parser.add_argument(
        "--input",
        default=None,
        help="Необязательный JSONL со входами Track A для сверки покрытия и ID evidence.",
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
