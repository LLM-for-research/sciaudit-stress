#!/usr/bin/env python3
"""Репетиция скрытой оценки (мануал §8.1, шаг 7; §10.5).

Седьмой шаг стандарта готовности: прогнать оценку по скрытому срезу целиком,
до того как скрытый срез появится по-настоящему. Репетиция проверяет не
качество системы, а **механику**: что штаб умеет нарезать скрытый срез, отдать
участникам ровно вход, получить предсказания, проверить их и посчитать метрики,
ни разу не показав gold.

Пять шагов, каждый со своим правом уронить репетицию:

1. нарезка по группам (§10.4) — статья и её стресс-варианты не расходятся;
2. **проверка студенческой половины** — в ней нет ни gold, ни приватных полей;
3. прогон бейзлайна по студенческому входу — ровно тем же CLI, каким сдают
   участники;
4. валидация предсказаний по схеме;
5. скоринг на штабной стороне и отчёт.

Второй шаг — единственная причина, по которой это не «просто ещё один прогон
evaluator'а». Проверяется отсутствие того, чего быть не должно, а такую ошибку
нельзя заметить по метрикам: если gold случайно уедет к участникам, метрики
станут только лучше.

Бейзлайн по умолчанию B0: он не ходит в модель, поэтому репетиция ничего не
стоит и гоняется в CI на каждом коммите.

Запуск::

    python -m sciaudit.evaluator.hidden_dry_run \\
        --internal data_paper_derived/*.internal_annotation.jsonl --by seed

Коды возврата: 0 — репетиция прошла, 1 — не прошла, 2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from sciaudit.baselines import b0_always_insufficient
from sciaudit.construction import make_hidden_split as splitter
from sciaudit.evaluator.score import score
from sciaudit.leakage.forbidden_key_scan import scan
from sciaudit.schemas.validate_predictions import validate_prediction_file

#: Поля, которых в студенческой половине быть не может ни под каким видом.
#: Список короче §10.1 намеренно: здесь ловится не забытый ключ внутри объекта,
#: а целый файл не с той стороны границы.
FORBIDDEN_FILES = ("gold.jsonl", "internal_annotation.jsonl")


def check_student_half(student_dir, split_name):
    """Убедиться, что участникам не уехало ничего, кроме входов (§10.5)."""
    problems = []
    path = Path(student_dir) / split_name
    for name in FORBIDDEN_FILES:
        if (path / name).exists():
            problems.append(f"{split_name}: в студенческой половине лежит {name}")

    leaks, _ = scan([path])
    problems.extend(f"{split_name}: {leak}" for leak in leaks)

    inputs = path / "inputs.jsonl"
    if not inputs.exists():
        problems.append(f"{split_name}: нет inputs.jsonl — участникам нечего отдать")
        return problems

    with open(inputs, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "gold" in row or "stress" in row or "review" in row:
                problems.append(
                    f"{split_name}: строка {number} несёт приватное поле")
    return problems


def dry_run(internal_paths, workdir, by="seed", ratios=None, split_name=None):
    """Прогнать репетицию. Возвращает ``(проблемы, отчёт)``."""
    ratios = ratios or splitter.DEFAULT_RATIOS
    workdir = Path(workdir)
    student_dir, staff_dir = workdir / "student", workdir / "staff"

    records = splitter.read_internal(internal_paths)
    if not records:
        raise ValueError("во входе нет записей")

    per_split, problems, paper_splits = splitter.split_records(records, by, ratios)
    for name, members in sorted(per_split.items()):
        splitter.write_split(name, members, student_dir, staff_dir)

    name = split_name or splitter.HIDDEN_VAL
    if name not in per_split:
        problems.append(f"репетировать нечего: сплит {name} не получился")
        return problems, {}

    problems.extend(check_student_half(student_dir, name))

    inputs = student_dir / name / "inputs.jsonl"
    predictions = workdir / f"{name}_predictions.jsonl"
    b0_always_insufficient.run(str(inputs), str(predictions))

    errors = validate_prediction_file(str(predictions), str(inputs))
    problems.extend(f"{name}: предсказание не по схеме — {error}" for error in errors)

    gold = staff_dir / name / "gold.jsonl"
    metrics = score(str(predictions), str(gold))
    if not metrics["submission"]["scoreable"]:
        problems.append(f"{name}: evaluator отказался скорить — "
                        f"{metrics['submission']['errors']}")

    report = {
        "split": name,
        "by": by,
        "instances": len(per_split[name]),
        "papers": sorted({splitter.paper_key(r) for r in per_split[name]}),
        "shared_papers": sorted(paper for paper, splits in paper_splits.items()
                                if len(splits) > 1),
        "accuracy": metrics["verdict"]["accuracy"],
        "coverage": metrics["coverage"]["value"],
        "sizes": {key: len(value) for key, value in sorted(per_split.items())},
    }
    return problems, report


def render(report):
    lines = [
        f"репетиция скрытой оценки на срезе {report['split']} "
        f"({report['instances']} инстанс(ов), статьи {', '.join(report['papers'])})",
        f"  нарезка по: {report['by']}; размеры сплитов: "
        + ", ".join(f"{k}={v}" for k, v in report["sizes"].items()),
        "  студенческая половина: только inputs.jsonl, приватных полей нет",
        f"  B0 отскорен штабом: accuracy {report['accuracy']:.3f}, "
        f"coverage {report['coverage']:.3f}",
    ]
    if report["shared_papers"]:
        lines.append("  ВНИМАНИЕ: статьи по обе стороны границы — "
                     + ", ".join(report["shared_papers"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Репетиция скрытой оценки от нарезки до метрик (§8.1).")
    parser.add_argument("--internal", nargs="+", required=True)
    parser.add_argument("--by", choices=("paper", "seed"), default="seed")
    parser.add_argument("--split", default=splitter.HIDDEN_VAL)
    parser.add_argument("--keep", default=None,
                        help="Куда сложить рабочую директорию. По умолчанию "
                             "временная: скрытые входы не должны оставаться "
                             "лежать в репозитории после репетиции.")
    args = parser.parse_args(argv)

    def execute(workdir):
        return dry_run(args.internal, workdir, by=args.by, split_name=args.split)

    try:
        if args.keep:
            Path(args.keep).mkdir(parents=True, exist_ok=True)
            problems, report = execute(args.keep)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                problems, report = execute(tmp)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if report:
        print("hidden_dry_run: " + render(report))
    if problems:
        for problem in problems:
            print(f"hidden_dry_run: {problem}", file=sys.stderr)
        print(f"hidden_dry_run: НЕ ПРОШЛА — {len(problems)} проблем(ы)")
        return 1

    print("hidden_dry_run: ПРОЙДЕНА — штаб может провести скрытую оценку, "
          "ни разу не показав gold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
