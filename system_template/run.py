#!/usr/bin/env python3
"""Шаблон системы SciAudit-Stress.

Заглушка каркаса: читает JSONL со входами и пишет ВАЛИДНЫЙ ПО СХЕМЕ JSONL с
предсказаниями (по одному на вход). Настоящего аудита она пока не делает — её
задача доказать, что рельса проходит от входа до конца («rails before trains»,
мануал §1.3, §8.1). Команды заменяют `audit_instance` своим пайплайном; внешний
контракт (--input / --output) обязан остаться прежним.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def audit_instance(instance: dict) -> dict:
    """Вернуть валидное по схеме предсказание для одного входного инстанса.

    Политика заглушки: всегда "insufficient" без evidence — повторяет
    санитарный бейзлайн B0 (мануал §11.2). Замените настоящей системой.
    """
    return {
        "instance_id": instance["instance_id"],
        "verdict": "insufficient",
        "confidence": 0.25,
        "predicted_eids": [],
        "issue_tags": ["evidence_missing_or_incomplete"],
        "abstain": False,
        "rationale_short": "Scaffold placeholder system; no real audit performed.",
        "runtime_seconds": 0.0,
        "estimated_cost": {"gpu_seconds": 0.0, "api_cost_usd": 0.0},
        "system_info": {
            "model": "scaffold-stub",
            "uses_numeric_checker": False,
            "uses_lora": False,
            "uses_vlm": False,
        },
    }


def run(input_path: str, output_path: str) -> int:
    n = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            start = time.perf_counter()
            pred = audit_instance(instance)
            pred["runtime_seconds"] = round(time.perf_counter() - start, 6)
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Раннер системы SciAudit-Stress.")
    parser.add_argument("--input", required=True, help="Путь к JSONL со входами.")
    parser.add_argument("--output", required=True, help="Путь для записи JSONL с предсказаниями.")
    args = parser.parse_args(argv)

    n = run(args.input, args.output)
    print(f"Записано предсказаний — {n}, файл {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
