#!/usr/bin/env python3
"""Проекция приватной аннотации в публичный сплит (мануал §1.4).

Публичный вход Track A получается из записи ``internal_annotation``
**вычёркиванием**, а не переписыванием: удаляются провенанс статьи, стресс-
метаданные, gold-метки и поля ревью, а текст claim и evidence остаётся ровно
тем же. Поэтому проекция обязана быть механической — иначе публичная и
приватная половины со временем разойдутся, и обсуждать станет нечего.

На выходе два файла рядом:

* ``inputs.jsonl`` — объекты ``schemas/track_a_input.schema.json``;
* ``gold.jsonl`` — ``instance_id`` плюс вердикт, supporting_eids и issue_tags.

``severity`` и ``private_rationale`` в публичный gold не попадают: они нужны
штабу для разбора спорных случаев, а системам — нет.

Запуск::

    python -m sciaudit.construction.derive_public \\
        --internal data_paper_derived/P001_libero_plus.internal_annotation.jsonl \\
                   data_paper_derived/P002_boom.internal_annotation.jsonl \\
        --out-dir data_public/public_dev

Коды возврата: 0 — успех, 2 — ошибка чтения или проверки.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sciaudit.schemas import FORBIDDEN_INPUT_KEYS, find_forbidden, read_jsonl

INPUT_SCHEMA_VERSION = "track_a_input_v1"

#: Поля единицы evidence, видимые участнику. Всё прочее (``source_ref``,
#: ``is_distractor``) — приватная разметка и вычёркивается.
PUBLIC_EVIDENCE_FIELDS = ("eid", "source_kind", "modality", "text", "normalized_numbers")

#: Поля claim, видимые участнику. ``claim_strength`` и ``source_location``
#: остаются приватными: первое подсказывает вердикт, второе — статью.
PUBLIC_CLAIM_FIELDS = ("text", "claim_type", "scope")


class DeriveError(Exception):
    """Запись нельзя спроецировать — это отказ, а не пропуск."""


def project_input(record):
    """Публичный вход Track A из приватной записи."""
    claim = record["claim"]
    pack = record["evidence_pack"]

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "paper_id": record["paper"]["paper_id"],
        "instance_id": record["instance_id"],
        "claim": {key: claim[key] for key in PUBLIC_CLAIM_FIELDS if key in claim},
        "evidence_pack": [
            {key: unit[key] for key in PUBLIC_EVIDENCE_FIELDS if key in unit}
            for unit in pack
        ],
        "allowed_evidence_ids": [unit["eid"] for unit in pack],
    }


def project_gold(record):
    """Публичная gold-строка: метка без обоснования и без severity."""
    gold = record["gold"]
    return {
        "instance_id": record["instance_id"],
        "gold": {
            "verdict": gold["verdict"],
            "supporting_eids": list(gold.get("supporting_eids", [])),
            "issue_tags": list(gold.get("issue_tags", [])),
        },
    }


def derive(internal_paths):
    """Вернуть ``(inputs, golds)`` по списку файлов приватных аннотаций."""
    inputs, golds, seen = [], [], set()

    for path in internal_paths:
        for line_no, record in read_jsonl(path):
            instance_id = record.get("instance_id")
            if not instance_id:
                raise DeriveError(f"{path}:{line_no}: нет instance_id")
            if instance_id in seen:
                raise DeriveError(f"{path}:{line_no}: дубликат instance_id {instance_id}")
            seen.add(instance_id)

            try:
                public = project_input(record)
                gold = project_gold(record)
            except KeyError as exc:
                raise DeriveError(f"{path}:{line_no}: нет обязательного поля {exc}") from exc

            leaks = find_forbidden(public, FORBIDDEN_INPUT_KEYS)
            if leaks:
                raise DeriveError(f"{path}:{line_no}: приватное поле в проекции: {leaks}")

            inputs.append(public)
            golds.append(gold)

    return inputs, golds


def write_jsonl(path, rows):
    Path(path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Спроецировать приватные аннотации в публичный сплит.")
    parser.add_argument("--internal", nargs="+", required=True,
                        help="JSONL с записями internal_annotation.")
    parser.add_argument("--out-dir", required=True, help="Куда положить inputs.jsonl и gold.jsonl.")
    args = parser.parse_args(argv)

    try:
        inputs, golds = derive([Path(p) for p in args.internal])
    except (DeriveError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "inputs.jsonl", inputs)
    write_jsonl(out_dir / "gold.jsonl", golds)

    print(f"derive_public: {len(inputs)} инстанс(ов) -> {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
