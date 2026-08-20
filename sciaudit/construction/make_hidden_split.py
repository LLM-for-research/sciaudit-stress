#!/usr/bin/env python3
"""Нарезка скрытых сплитов из приватных аннотаций (мануал §10.4, §10.5).

Скрытый сплит — не «часть данных, которую мы никому не показали», а часть,
которую **нельзя вывести** из показанного. Поэтому нарезка идёт не по
инстансам, а по группам (§10.4):

* **группа статьи** — все инстансы одной публикации. Два инстанса из одной
  статьи делят таблицы, числа и терминологию; разложив их по разные стороны,
  штаб меряет не обобщение, а память о конкретной таблице.
* **группа seed** — исходный инстанс вместе со всеми его стресс-вариантами.
  Стресс-вариант отличается от seed одним контролируемым изменением, так что
  вариант в скрытом срезе при публичном seed — это публичная подсказка.

Обе группы соблюдаются одновременно: сначала инстансы собираются в группы seed,
затем группы seed — в группы статей, и в сплит уходит группа целиком.

Что получается на выходе — две половины, и это главное свойство модуля:

* **студенческая** (``--student-out``): для скрытых срезов только
  ``inputs.jsonl``, без gold. Так требует §10.5: участники получают вход и
  ничего больше;
* **штабная** (``--staff-out``): ``gold.jsonl`` для скоринга.

Проекция в публичный вид переиспользуется из
:mod:`sciaudit.construction.derive_public` — вычёркивание приватных полей
обязано иметь ровно одну реализацию, иначе однажды разойдётся.

Запуск::

    python -m sciaudit.construction.make_hidden_split \\
        --internal data_paper_derived/*.internal_annotation.jsonl \\
        --by paper --student-out /tmp/student --staff-out /tmp/staff

Коды возврата: 0 — нарезано, 1 — гарантии сплита не выполняются,
2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sciaudit.construction.derive_public import (
    project_gold,
    project_input,
    write_jsonl,
)

#: Имена сплитов. Публичный — единственный, чей gold уходит студентам.
PUBLIC_SPLIT = "public"
HIDDEN_VAL = "hidden_val"
HIDDEN_TEST = "hidden_test"
SPLITS = (PUBLIC_SPLIT, HIDDEN_VAL, HIDDEN_TEST)

#: Доли инстансов по умолчанию. Ориентир §8.3: скрытая валидация 40–60,
#: финальный скрытый резерв 30–40 при public-dev 80–150.
DEFAULT_RATIOS = {PUBLIC_SPLIT: 0.5, HIDDEN_VAL: 0.3, HIDDEN_TEST: 0.2}


def read_internal(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def seed_key(record):
    """Инстанс и все его стресс-варианты имеют один ключ."""
    stress = record.get("stress") or {}
    return stress.get("seed_instance_id") or record["instance_id"]


def paper_key(record):
    return (record.get("paper") or {}).get("paper_id") or "UNKNOWN"


def build_groups(records, by):
    """Собрать инстансы в неделимые группы.

    ``by="seed"`` — группа это seed со своими вариантами; ``by="paper"`` —
    статья целиком. Паперная нарезка строже: она включает в себя seed-группы,
    потому что варианты одного seed всегда из одной статьи.
    """
    if by not in ("paper", "seed"):
        raise ValueError("--by принимает только paper или seed")

    key = paper_key if by == "paper" else seed_key
    groups = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return dict(groups)


def assign_groups(groups, ratios):
    """Разложить группы по сплитам, приближая заданные доли инстансов.

    Жадно и детерминированно: большие группы первыми, каждая уходит в сплит,
    который дальше всех от своей цели. Случайности здесь не нужно — при
    десятках инстансов она только делает результат невоспроизводимым.
    """
    total = sum(len(members) for members in groups.values())
    targets = {name: ratio * total for name, ratio in ratios.items() if ratio > 0}
    if not targets:
        raise ValueError("все доли нулевые — нечего нарезать")

    filled = {name: 0 for name in targets}
    assignment = {}
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), str(item[0])))
    for name, members in ordered:
        split = min(targets, key=lambda s: (filled[s] - targets[s], s))
        assignment[name] = split
        filled[split] += len(members)
    return assignment


def check_guarantees(groups, assignment, records, requested=()):
    """Что обязано выполняться после нарезки. Пустой список — всё в порядке.

    ``requested`` — сплиты, которые заказывали ненулевой долей. Пустой сплит из
    этого списка это не мелочь, а сообщение о том, что групп меньше, чем
    сплитов: разложить 2 статьи на 3 группы-непересекающиеся части нельзя, и
    молча отдать две вместо трёх значило бы соврать про скрытый резерв.
    """
    problems = []

    by_instance = {}
    for name, members in groups.items():
        for record in members:
            instance_id = record["instance_id"]
            if instance_id in by_instance:
                problems.append(f"{instance_id}: инстанс попал в две группы")
            by_instance[instance_id] = assignment[name]

    seed_splits = defaultdict(set)
    paper_splits = defaultdict(set)
    for record in records:
        split = by_instance.get(record["instance_id"])
        seed_splits[seed_key(record)].add(split)
        paper_splits[paper_key(record)].add(split)

    for key, splits in sorted(seed_splits.items()):
        if len(splits) > 1:
            problems.append(
                f"seed-группа {key} разошлась по сплитам {sorted(splits)}: "
                "стресс-вариант скрытого инстанса лежит в публичном срезе")

    filled = set(by_instance.values())
    empty = [name for name in requested if name not in filled]
    problems.extend(
        f"сплит {name} заказан, но пуст: неделимых групп меньше, чем сплитов"
        for name in empty)
    return problems, paper_splits


def split_records(records, by, ratios):
    groups = build_groups(records, by)
    assignment = assign_groups(groups, ratios)
    requested = [name for name in SPLITS if ratios.get(name, 0) > 0]
    problems, paper_splits = check_guarantees(groups, assignment, records, requested)

    per_split = defaultdict(list)
    for name, members in groups.items():
        per_split[assignment[name]].extend(members)
    for members in per_split.values():
        members.sort(key=lambda record: record["instance_id"])
    return dict(per_split), problems, paper_splits


def write_split(name, records, student_dir, staff_dir):
    """Разложить сплит по двум половинам. Скрытый gold студентам не уходит."""
    student_path = Path(student_dir) / name
    staff_path = Path(staff_dir) / name
    student_path.mkdir(parents=True, exist_ok=True)
    staff_path.mkdir(parents=True, exist_ok=True)

    write_jsonl(student_path / "inputs.jsonl", [project_input(r) for r in records])
    golds = [project_gold(r) for r in records]
    write_jsonl(staff_path / "gold.jsonl", golds)
    if name == PUBLIC_SPLIT:
        write_jsonl(student_path / "gold.jsonl", golds)
    return student_path, staff_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Нарезать скрытые сплиты по группам статей и seed (§10.4).")
    parser.add_argument("--internal", nargs="+", required=True)
    parser.add_argument("--by", choices=("paper", "seed"), default="paper",
                        help="Единица нарезки. paper строже и требуется §10.4; "
                             "seed допустим, пока статей мало, но тогда одна "
                             "статья окажется по обе стороны.")
    parser.add_argument("--student-out", required=True)
    parser.add_argument("--staff-out", required=True)
    parser.add_argument("--public", type=float, default=DEFAULT_RATIOS[PUBLIC_SPLIT])
    parser.add_argument("--hidden-val", type=float, default=DEFAULT_RATIOS[HIDDEN_VAL])
    parser.add_argument("--hidden-test", type=float, default=DEFAULT_RATIOS[HIDDEN_TEST])
    args = parser.parse_args(argv)

    ratios = {PUBLIC_SPLIT: args.public, HIDDEN_VAL: args.hidden_val,
              HIDDEN_TEST: args.hidden_test}

    try:
        records = read_internal(args.internal)
        if not records:
            raise ValueError("во входе нет записей")
        per_split, problems, paper_splits = split_records(records, args.by, ratios)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for name in SPLITS:
        if name in per_split:
            student_path, staff_path = write_split(
                name, per_split[name], args.student_out, args.staff_out)
            papers = sorted({paper_key(r) for r in per_split[name]})
            print(f"make_hidden_split: {name} — {len(per_split[name])} инстанс(ов), "
                  f"статьи {', '.join(papers)}; вход {student_path}, gold {staff_path}")

    shared = sorted(paper for paper, splits in paper_splits.items() if len(splits) > 1)
    if shared:
        print("make_hidden_split: ВНИМАНИЕ — статьи по обе стороны границы: "
              + ", ".join(shared))
        print("  §10.4 требует нарезки по группам статей. Пока статей мало, это "
              "неизбежно, но скрытый срез тогда меряет обобщение слабее, чем "
              "заявлено: часть таблиц системе уже знакома.")

    if problems:
        for problem in problems:
            print(f"make_hidden_split: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
