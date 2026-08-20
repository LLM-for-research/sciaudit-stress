#!/usr/bin/env python3
"""Сравнение бейзлайнов на одном срезе с повторами и разбросом (мануал §11.4).

Харнесс гоняет выбранные бейзлайны на одном входе, скорит каждый прогон
существующим evaluator'ом и печатает markdown-таблицу. Бейзлайны различаются
ровно одним элементом каждый (:mod:`sciaudit.baselines.model_audit`), поэтому
разница в метриках относится к этому элементу, а не к промпту или парсингу:

* B1 против B2 — вклад ретрива;
* B2 против B3 — вклад детерминированного инструмента;
* B3 против B4 — цена осторожности.

**Повторы обязательны, а не желательны.** На облачном эндпоинте при
``temperature: 0.0`` два одинаковых прогона B2 разошлись в 6 вердиктах из 24, и
одна ячейка таблицы сдвинулась на 16.7 пункта — больше, чем разрыв между
сравниваемыми системами. Поэтому ``--repeats N`` гоняет каждую конфигурацию N
раз, в таблицу идёт медиана с разбросом, а вывод о порядке систем печатается
отдельной строкой — вместе с честным «разброс этого не позволяет», если
победитель менялся между прогонами.

Запуск::

    python -m sciaudit.baselines.compare_baselines \\
        --input data_public/public_dev/inputs.jsonl \\
        --gold data_public/public_dev/gold.jsonl \\
        --systems b1 b2 --repeats 3 --model-api \\
        --out docs/baselines_compared.md

Без ``--model-api`` и ``--model-command`` берётся детерминированная заглушка
:mod:`sciaudit.baselines.stub_model`. Числа с заглушкой характеризуют харнесс,
а не качество аудита, и таблица помечается соответствующим предупреждением.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

from sciaudit.baselines import b1_bm25_llm, b2_fullpack_llm, b3_checked_llm, b4_selective
from sciaudit.baselines.model_audit import count_fallbacks, resolve_model
from sciaudit.evaluator.score import score

STUB_COMMAND = f"{sys.executable} -m sciaudit.baselines.stub_model"
STUB_MODEL_NAME = "deterministic-stub-not-a-model"
DEFAULT_TOP_KS = (1, 2, 3)
DEFAULT_SYSTEMS = ("b1", "b2")
KNOWN_SYSTEMS = ("b1", "b2", "b3", "b4")

#: Метрики, по которым считается медиана и разброс.
METRICS = ("accuracy", "macro_f1", "evidence_f1", "sfwr", "coverage", "augrc")


def _row(name, metrics, predictions):
    if not metrics["submission"]["scoreable"]:
        raise ValueError(
            f"{name}: evaluator отказался скорить сабмишен — "
            f"{metrics['submission']['errors']}. Бейзлайн обязан выдавать ровно одно "
            "предсказание на каждый gold-инстанс."
        )
    return {
        "system": name,
        "accuracy": metrics["verdict"]["accuracy"],
        "macro_f1": metrics["verdict"]["macro_f1"],
        "evidence_f1": metrics["evidence"]["f1"],
        "sfwr": metrics["severe_false_warrant_rate_non_abstained"]["rate"],
        "coverage": metrics["coverage"]["value"],
        "augrc": metrics["selective_risk"]["augrc"],
        "answered": metrics["counts"]["non_abstained_predictions"],
        "total": metrics["counts"]["gold_instances"],
        "fallbacks": count_fallbacks(predictions),
    }


def _plan(systems, top_ks, votes, confidence_threshold):
    """Что именно прогонять: пары «имя строки — как получить предсказания»."""
    plan = []
    for name in systems:
        if name == "b1":
            for top_k in top_ks:
                plan.append((
                    f"B1 (BM25, top-k={top_k})", f"b1_top{top_k}",
                    lambda inp, out, k=top_k, **kw: b1_bm25_llm.run(inp, out, top_k=k, **kw)))
        elif name == "b2":
            plan.append(("B2 (весь пак)", "b2", b2_fullpack_llm.run))
        elif name == "b3":
            plan.append(("B3 (пак + чекер)", "b3", b3_checked_llm.run))
        elif name == "b4":
            plan.append((
                f"B4 (отказ, votes={votes})", "b4",
                lambda inp, out, **kw: b4_selective.run(
                    inp, out, votes=votes,
                    confidence_threshold=confidence_threshold, **kw)))
        else:
            raise ValueError(f"неизвестный бейзлайн {name!r}; известны: "
                             + ", ".join(KNOWN_SYSTEMS))
    return plan


def _one_pass(plan, input_path, gold_path, common, tmp, tag):
    rows = []
    for name, slug, runner in plan:
        out = tmp / f"{slug}_{tag}.jsonl"
        predictions = runner(input_path, out, **common)
        rows.append(_row(name, score(out, gold_path), predictions))
    return rows


def _aggregate(passes):
    """Медиана по повторам плюс сами значения — разброс считается по ним."""
    order = [r["system"] for r in passes[0]]
    by_system = {name: [] for name in order}
    for rows in passes:
        for row in rows:
            by_system[row["system"]].append(row)

    aggregated = []
    for name in order:
        rows = by_system[name]
        agg = {"system": name, "repeats": len(rows), "values": {}}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            agg[metric] = statistics.median(values)
            agg["values"][metric] = values
        agg["answered"] = int(round(statistics.median(r["answered"] for r in rows)))
        agg["total"] = rows[0]["total"]
        agg["fallbacks"] = sum(r["fallbacks"] for r in rows)
        aggregated.append(agg)
    return aggregated


def stability(rows):
    """Устойчив ли порядок систем по accuracy между повторами.

    Возвращает ``(устойчив, победители по прогонам)``. Это и есть ответ на
    вопрос, ради которого делаются повторы: если победитель меняется, таблица
    не разрешает сравнение систем — и обязана сказать это прямо.
    """
    repeats = rows[0]["repeats"] if rows else 0
    winners = []
    for i in range(repeats):
        best = max(rows, key=lambda r: r["values"]["accuracy"][i])
        winners.append(best["system"])
    return len(set(winners)) == 1, winners


def compare(input_path, gold_path, model_command, model_name, top_ks=DEFAULT_TOP_KS,
            workdir=None, model_fn=None, systems=DEFAULT_SYSTEMS, repeats=1,
            votes=b4_selective.DEFAULT_VOTES,
            confidence_threshold=b4_selective.DEFAULT_CONFIDENCE_THRESHOLD):
    """Прогнать выбранные бейзлайны ``repeats`` раз, вернуть строки таблицы."""
    if repeats < 1:
        raise ValueError("--repeats должно быть не меньше 1.")
    plan = _plan(systems, top_ks, votes, confidence_threshold)
    common = {"model_command": model_command, "model_name": model_name,
              "model_fn": model_fn}

    passes = []
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        for repeat in range(repeats):
            passes.append(_one_pass(plan, input_path, gold_path, common, tmp,
                                    tag=f"r{repeat}"))
    return _aggregate(passes)


def _cell(row, metric):
    values = row["values"][metric] if "values" in row else [row[metric]]
    text = f"{row[metric]:.3f}"
    if len(values) > 1 and max(values) - min(values) > 1e-9:
        text += f" [{min(values):.3f}–{max(values):.3f}]"
    return text


def to_markdown(rows, model_name, input_path, gold_path, is_stub,
                transport="--model-api", top_ks=DEFAULT_TOP_KS,
                systems=DEFAULT_SYSTEMS, repeats=1):
    retrieval_only = tuple(systems) == ("b1", "b2")
    title = ("B1 против B2: помогает ли ретрив" if retrieval_only
             else "Сравнение бейзлайнов")
    lines = [
        f"# {title}",
        "",
        "B1 подаёт модели top-k единиц evidence по BM25, B2 — весь evidence pack,",
        "B3 добавляет к тому же паку детерминированную проверку чисел, B4 — правила",
        "отказа поверх ансамбля голосов. Общий слой вызова модели у них один",
        "(`sciaudit/baselines/model_audit.py`), поэтому разница в метриках относится",
        "к тому единственному, чем бейзлайны различаются, а не к промпту, разбору",
        "ответа или обработке отказов.",
        "",
    ]

    if is_stub:
        lines += [
            "> **Числа ниже получены детерминированной заглушкой, а не моделью.**",
            "> Заглушка (`sciaudit/baselines/stub_model.py`) отвечает по трём жёстким",
            "> правилам и ничего не измеряет. Таблица показывает, что харнесс сравнения",
            "> работает и что бюджет ретрива действительно меняет результат. Качеством",
            "> аудита эти числа не являются и в отчёты о системах не идут.",
            "",
        ]

    lines += [
        f"- Вход: `{input_path}`",
        f"- Gold: `{gold_path}`",
        f"- Идентификатор модели: `{model_name}`",
        f"- Повторов каждой конфигурации: **{repeats}**",
        "",
    ]

    if repeats > 1:
        lines += ["В ячейках медиана по повторам, в скобках — наблюдённый разброс "
                  "(минимум–максимум).", ""]

    lines += [
        "| Система | Accuracy | Macro-F1 | Evidence F1 | SFWR | Coverage | AUGRC | Отказов парсинга |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {_cell(row, 'accuracy')} | {_cell(row, 'macro_f1')} | "
            f"{_cell(row, 'evidence_f1')} | {_cell(row, 'sfwr')} | "
            f"{_cell(row, 'coverage')} ({row['answered']}/{row['total']}) | "
            f"{_cell(row, 'augrc')} | {row['fallbacks']} |"
        )
    lines.append("")

    if repeats > 1 and rows:
        stable, winners = stability(rows)
        if stable:
            lines += [
                f"> **Порядок систем устойчив.** В каждом из {repeats} прогонов лучшей",
                f"> по accuracy оказалась одна и та же система — {winners[0]}. Вывод",
                "> сравнения этими повторами поддержан.",
                "",
            ]
        else:
            order = " → ".join(winners)
            lines += [
                f"> **Разброс не позволяет назвать победителя.** По прогонам лучшей",
                f"> оказывалась то одна система, то другая: {order}. Это свойство не",
                "> кода, а эндпоинта: недетерминизм остаётся при `temperature: 0.0` и",
                "> `seed`. Пока порядок не устойчив, из таблицы следует только то, что",
                "> харнесс работает, — но не то, какая система лучше.",
                "",
            ]
    else:
        lines += [
            "> **Из одного прогона этой таблицы вывод не следует.** Два одинаковых",
            "> прогона этого же скрипта при `temperature: 0.0` дали противоположный",
            "> ответ: в первом лучшим оказался B1 с top-k=3 (accuracy 0.667 против",
            "> 0.625 у B2), во втором — B2 (0.708 против 0.500). Одна ячейка сдвинулась",
            "> на 16.7 пункта. Причина не в коде: два прогона B2 на одном входе",
            "> разошлись в 6 вердиктах из 24, а `seed` разброс уменьшает, но не",
            "> снимает — на облачном инференсе недетерминизм принадлежит эндпоинту,",
            "> а не параметрам запроса.",
            ">",
            "> Практический вывод: пока каждая конфигурация не прогнана несколько раз",
            "> и в таблице не стоит разброс, эти числа показывают, что харнесс",
            "> работает, и не показывают, помогает ли ретрив. Запустите тот же",
            "> скрипт с `--repeats 3`.",
            "",
        ]

    lines += [
        "Accuracy и macro-F1 считаются только по инстансам без отказа, цена отказа",
        "видна в колонке coverage; SFWR — доля severe-обоснований среди отвеченных.",
        "AUGRC — площадь под кривой обобщённого риска: она строится ранжированием по",
        "уверенности, поэтому массовый отказ и одинаковая уверенность её не улучшают.",
        "Колонка «отказов парсинга» показывает, сколько раз ответ модели не",
        "разобрался и подставился безопасный отказ (суммарно по всем повторам): если",
        "она не нулевая, метрики читать нельзя.",
        "",
        "## Как перегенерировать",
        "",
        "```bash",
        "python -m sciaudit.baselines.compare_baselines \\",
        f"  --input {input_path} \\",
        f"  --gold {gold_path} \\",
        f"  {transport} \\",
        f"  --systems {' '.join(systems)} \\",
        f"  --top-k {top_ks} \\",
        f"  --repeats {repeats} \\",
        "  --out docs/baselines_compared.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Сравнение бейзлайнов на одном срезе с повторами.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--model-command", default=None,
                        help="Команда модели. По умолчанию — детерминированная заглушка.")
    parser.add_argument("--model-api", action="store_true",
                        help="Вызывать модель по OpenAI-совместимому API "
                             "(SCIAUDIT_BASE_URL / SCIAUDIT_MODEL / SCIAUDIT_API_KEY).")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS),
                        help="Какие бейзлайны сравнивать: " + " ".join(KNOWN_SYSTEMS))
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_KS),
                        help="Бюджеты ретрива для B1.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Сколько раз прогнать каждую конфигурацию. Один прогон "
                             "сравнения систем не обосновывает — см. разброс в шапке.")
    parser.add_argument("--votes", type=int, default=b4_selective.DEFAULT_VOTES,
                        help="Голосов на инстанс у B4.")
    parser.add_argument("--confidence-threshold", type=float,
                        default=b4_selective.DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Порог уверенности у B4.")
    parser.add_argument("--out", default=None, help="Куда записать markdown-заметку.")
    args = parser.parse_args(argv)

    is_stub = args.model_command is None and not args.model_api
    if is_stub:
        # Заглушка — это тоже «команда модели», поэтому она проходит тем же путём.
        args.model_command = STUB_COMMAND
        args.model_name = args.model_name or STUB_MODEL_NAME
    else:
        args.model_name = args.model_name or "open-model"

    try:
        model_fn, model_command, model_name = resolve_model(args)
        rows = compare(args.input, args.gold, model_command, model_name,
                       tuple(args.top_k), model_fn=model_fn,
                       systems=tuple(args.systems), repeats=args.repeats,
                       votes=args.votes,
                       confidence_threshold=args.confidence_threshold)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    transport = "--model-api" if args.model_api else (
        "" if is_stub else f'--model-command "{args.model_command}"')
    markdown = to_markdown(rows, model_name, args.input, args.gold, is_stub,
                           transport=transport,
                           top_ks=" ".join(str(k) for k in args.top_k),
                           systems=tuple(args.systems), repeats=args.repeats)

    if args.out:
        Path(args.out).write_text(markdown + "\n", encoding="utf-8")
        print(f"compare_baselines: записано в {args.out}", file=sys.stderr)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
