#!/usr/bin/env python3
"""B0 — бейзлайн «всегда insufficient» (мануал §11.2).

Тривиальная система, валидная по формату: всегда возвращает ``insufficient`` без
единого evidence. Её задача — проверять работоспособность схем и evaluator'а;
это научная нижняя граница, которую обязана воспроизвести каждая команда, прежде
чем заявлять об улучшении (§11.1). Ни модели, ни retrieval — только стандартная
библиотека Python.

Запуск:
    python -m sciaudit.baselines.b0_always_insufficient --input in.jsonl --output out.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys


def predict(instance: dict) -> dict:
    """Вернуть тривиальное предсказание «всегда insufficient» для одного инстанса."""
    return {
        "instance_id": instance["instance_id"],
        "verdict": "insufficient",
        "confidence": 0.25,
        "predicted_eids": [],
        "issue_tags": ["evidence_missing_or_incomplete"],
        "abstain": False,
        "rationale_short": "Trivial baseline.",
        "runtime_seconds": 0.0,
        "estimated_cost": {"gpu_seconds": 0.0, "api_cost_usd": 0.0},
        "system_info": {
            "model": "none",
            "uses_numeric_checker": False,
            "uses_lora": False,
            "uses_vlm": False,
        },
    }


def run(input_path: str, output_path: str) -> int:
    """Прочитать JSONL со входами и записать по предсказанию на строку. Вернуть их число."""
    n = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            fout.write(json.dumps(predict(instance), ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Бейзлайн B0 «всегда insufficient».")
    parser.add_argument("--input", required=True, help="Путь к JSONL со входами.")
    parser.add_argument("--output", required=True, help="Путь для записи JSONL с предсказаниями.")
    args = parser.parse_args(argv)
    n = run(args.input, args.output)
    print(f"B0: записано предсказаний — {n}, файл {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
