#!/usr/bin/env python3
"""Сравнение B1 и B2: помогает ли ретрив (issue #14, мануал §11.4).

B1 и B2 отличаются ровно одним — B1 подаёт модели top-k единиц evidence по
BM25, B2 весь пак. Всё остальное общее (:mod:`sciaudit.baselines.model_audit`),
поэтому разница в метриках относится к ретриву, а не к промпту или парсингу.

Скрипт гоняет B2 один раз и B1 на нескольких бюджетах top-k, скорит каждый
прогон существующим evaluator'ом и печатает markdown-таблицу.

Запуск::

    python -m sciaudit.baselines.compare_b1_b2 \\
        --input data_public/public_warmup/inputs.jsonl \\
        --gold data_public/public_warmup/gold.jsonl \\
        --model-command "your-open-model-command" \\
        --model-name "your-open-model-id" \\
        --out docs/b1_vs_b2.md

Без ``--model-command`` берётся детерминированная заглушка
:mod:`sciaudit.baselines.stub_model`. Числа с заглушкой характеризуют харнесс,
а не качество аудита, и таблица помечается соответствующим предупреждением.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from sciaudit.baselines import b1_bm25_llm, b2_fullpack_llm
from sciaudit.baselines.model_audit import count_fallbacks, resolve_model
from sciaudit.evaluator.score import score

STUB_COMMAND = f"{sys.executable} -m sciaudit.baselines.stub_model"
STUB_MODEL_NAME = "deterministic-stub-not-a-model"
DEFAULT_TOP_KS = (1, 2, 3)


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


def compare(input_path, gold_path, model_command, model_name, top_ks=DEFAULT_TOP_KS,
            workdir=None, model_fn=None):
    """Прогнать B2 и B1 на каждом top-k, вернуть список строк таблицы."""
    rows = []
    common = {"model_command": model_command, "model_name": model_name,
              "model_fn": model_fn}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)

        for top_k in top_ks:
            out = tmp / f"b1_top{top_k}.jsonl"
            preds = b1_bm25_llm.run(input_path, out, top_k=top_k, **common)
            rows.append(_row(f"B1 (BM25, top-k={top_k})", score(out, gold_path), preds))

        out = tmp / "b2.jsonl"
        preds = b2_fullpack_llm.run(input_path, out, **common)
        rows.append(_row("B2 (весь пак)", score(out, gold_path), preds))

    return rows


def to_markdown(rows, model_name, input_path, gold_path, is_stub,
                transport="--model-api", top_ks=DEFAULT_TOP_KS):
    lines = [
        "# B1 против B2: помогает ли ретрив",
        "",
        "B1 подаёт модели top-k единиц evidence по BM25, B2 — весь evidence pack.",
        "Общий слой вызова модели у них один (`sciaudit/baselines/model_audit.py`),",
        "поэтому разница в метриках относится к ретриву, а не к промпту, разбору",
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
            "> Настоящие значения появятся, когда закроется issue #12 (доступ к модели).",
            "",
        ]

    lines += [
        f"- Вход: `{input_path}`",
        f"- Gold: `{gold_path}`",
        f"- Идентификатор модели: `{model_name}`",
        "",
        "| Система | Accuracy | Macro-F1 | Evidence F1 | SFWR | Coverage | AUGRC | Отказов парсинга |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        lines.append(
            f"| {r['system']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
            f"{r['evidence_f1']:.3f} | {r['sfwr']:.3f} | "
            f"{r['coverage']:.3f} ({r['answered']}/{r['total']}) | "
            f"{r['augrc']:.3f} | {r['fallbacks']} |"
        )

    lines += [
        "",
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
        "> работает, и не показывают, помогает ли ретрив. Лидербордное сравнение",
        "> систем на таком эндпоинте требует того же — иначе места распределит шум.",
        "",
        "Accuracy и macro-F1 считаются только по инстансам без отказа, цена отказа",
        "видна в колонке coverage; SFWR — доля severe-обоснований среди отвеченных.",
        "AUGRC — площадь под кривой обобщённого риска: она строится ранжированием по",
        "уверенности, поэтому массовый отказ и одинаковая уверенность её не улучшают.",
        "Колонка «отказов парсинга» показывает, сколько раз ответ модели не",
        "разобрался и подставился безопасный отказ: если",
        "она не нулевая, метрики читать нельзя.",
        "",
        "## Как перегенерировать",
        "",
        "```bash",
        "python -m sciaudit.baselines.compare_b1_b2 \\",
        f"  --input {input_path} \\",
        f"  --gold {gold_path} \\",
        f"  {transport} \\",
        f"  --top-k {top_ks} \\",
        "  --out docs/b1_vs_b2.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение B1 и B2 на одном срезе.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--model-command", default=None,
                        help="Команда модели. По умолчанию — детерминированная заглушка.")
    parser.add_argument("--model-api", action="store_true",
                        help="Вызывать модель по OpenAI-совместимому API "
                             "(SCIAUDIT_BASE_URL / SCIAUDIT_MODEL / SCIAUDIT_API_KEY).")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_KS))
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
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows = compare(args.input, args.gold, model_command, model_name,
                   tuple(args.top_k), model_fn=model_fn)
    transport = "--model-api" if args.model_api else (
        "" if is_stub else f'--model-command "{args.model_command}"')
    markdown = to_markdown(rows, model_name, args.input, args.gold, is_stub,
                           transport=transport,
                           top_ks=" ".join(str(k) for k in args.top_k))

    if args.out:
        Path(args.out).write_text(markdown + "\n", encoding="utf-8")
        print(f"compare_b1_b2: записано в {args.out}", file=sys.stderr)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
