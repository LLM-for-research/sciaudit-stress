#!/usr/bin/env python3
"""Аудит пересечения сплитов (мануал §10.4).

Один и тот же инстанс, попавший в два сплита, ломает измерение: система,
настроенная на примерах, получает бесплатные очки на warm-up, а warm-up
перестаёт быть независимым замером. Совпадение при этом почти никогда не
буквальное — обычно это перефразированный claim под новым `instance_id`,
поэтому сравниваются не хеши, а нормализованные тексты.

Что считается пересечением:

* **id_collision** — один `instance_id` встречается в двух сплитах;
* **cross_split_claim** — claim из разных сплитов похожи не меньше порога;
* **intra_split_duplicate** — внутри одного сплита два claim совпадают
  дословно (после нормализации), то есть инстанс скопировали.

Чем пересечение НЕ является — два случая, на которых проверка обязана молчать:

* общий evidence pack при разных claim: это осознанный контроль (минимальная
  пара warranted/overclaimed на одном паке);
* почти одинаковые claim *внутри* одного сплита: ровно так выглядит стресс-пара
  seed → усиленный claim (§5.4). Поэтому внутри сплита ловится только точное
  совпадение, а порог похожести применяется исключительно между сплитами.

Файлы-иллюстрации схем (`*internal_annotation*`) сплитами не являются и в аудит
не попадают — то же исключение, что и в ``forbidden_key_scan``.

Запуск::

    python -m sciaudit.leakage.split_overlap_check examples data_public/public_warmup
    python -m sciaudit.leakage.split_overlap_check data_public/* --threshold 0.9

Коды возврата: 0 — чисто, 1 — найдено пересечение, 2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

#: Порог похожести claim, выше которого два инстанса считаются одним.
#: 0.85 выбран с запасом: внутри текущего warm-up максимум по непохожим парам
#: около 0.66, а намеренная минимальная пара на общем паке даёт 0.34.
DEFAULT_THRESHOLD = 0.85

_WS = re.compile(r"\s+")

#: Имена, которые не являются сплитами: примеры приватной схемы. Список тот же,
#: что и `internal`-профиль в ``forbidden_key_scan``.
NON_SPLIT_PATTERNS = ("internal_annotation",)


class OverlapError(Exception):
    """Файл не читается или аргументы некорректны — это отказ, а не пропуск."""


def normalize(text: str) -> str:
    """Регистр и пробелы не должны прятать совпадение."""
    return _WS.sub(" ", text.strip().lower())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def load_instances(path: Path) -> list[dict]:
    """Собрать инстансы Track A из одного JSONL.

    Файлы gold, предсказаний и манифестов пропускаются: инстансом считается
    только объект, у которого есть и `claim`, и `evidence_pack`.
    """
    instances = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise OverlapError(f"{path}: не читается: {e}") from e

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise OverlapError(f"{path}:{line_no}: битый JSON: {e}") from e
        if not isinstance(obj, dict):
            continue
        claim = obj.get("claim")
        if not isinstance(claim, dict) or "evidence_pack" not in obj:
            continue
        claim_text = claim.get("text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            continue
        instances.append({
            "instance_id": obj.get("instance_id", f"<{path.name}:{line_no}>"),
            "claim": claim_text,
            "where": f"{path}:{line_no}",
        })
    return instances


def collect_split(target: Path) -> list[dict]:
    """Все инстансы одного сплита. Директория обходится рекурсивно."""
    if not target.exists():
        raise OverlapError(f"{target}: нет такого файла или директории")
    files = [target] if target.is_file() else sorted(target.rglob("*.jsonl"))
    instances = []
    for f in files:
        if any(p in f.name for p in NON_SPLIT_PATTERNS):
            continue
        instances.extend(load_instances(f))
    return instances


def find_overlaps(splits: dict[str, list[dict]], threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    """Сравнить сплиты попарно и внутри себя. Возвращает описания находок."""
    problems: list[str] = []
    names = sorted(splits)

    # --- между сплитами -------------------------------------------------
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left, right = splits[left_name], splits[right_name]

            shared_ids = {x["instance_id"] for x in left} & {y["instance_id"] for y in right}
            for instance_id in sorted(shared_ids):
                problems.append(
                    f"id_collision: {instance_id} присутствует и в {left_name}, и в {right_name}"
                )

            for x in left:
                for y in right:
                    ratio = similarity(x["claim"], y["claim"])
                    if ratio >= threshold:
                        problems.append(
                            f"cross_split_claim: {x['instance_id']} ({left_name}) ~ "
                            f"{y['instance_id']} ({right_name}) — сходство {ratio:.2f}\n"
                            f"    {x['where']}: {x['claim'][:90]}\n"
                            f"    {y['where']}: {y['claim'][:90]}"
                        )

    # --- внутри сплита ----------------------------------------------------
    # Только дословное совпадение: почти одинаковые claim здесь легальны, это
    # стресс-пара seed → усиленный claim.
    for name in names:
        items = splits[name]
        for i, x in enumerate(items):
            for y in items[i + 1:]:
                if normalize(x["claim"]) == normalize(y["claim"]):
                    problems.append(
                        f"intra_split_duplicate: {x['instance_id']} и {y['instance_id']} "
                        f"в {name} несут дословно один claim"
                    )

    return problems


def check(targets: list[Path], threshold: float = DEFAULT_THRESHOLD) -> tuple[list[str], dict[str, int]]:
    splits = {str(t): collect_split(t) for t in targets}
    return find_overlaps(splits, threshold), {k: len(v) for k, v in splits.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Аудит сплитов на дублирующиеся инстансы (мануал §10.4).",
    )
    parser.add_argument("paths", nargs="+", help="Директории сплитов или файлы JSONL для сравнения.")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Порог похожести claim, 0..1 (по умолчанию {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.threshold <= 1.0:
        print("ERROR: --threshold должен быть в (0, 1]", file=sys.stderr)
        return 2

    try:
        problems, sizes = check([Path(p) for p in args.paths], args.threshold)
    except OverlapError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    for name in sorted(sizes):
        print(f"  {sizes[name]:>4} инстанс(ов)  {name}")

    if problems:
        for p in problems:
            print(f"OVERLAP: {p}")
        print(f"split_overlap_check: FAIL — {len(problems)} пересечение(й)")
        return 1

    total = sum(sizes.values())
    print(f"split_overlap_check: OK — {total} инстанс(ов), пересечений нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
