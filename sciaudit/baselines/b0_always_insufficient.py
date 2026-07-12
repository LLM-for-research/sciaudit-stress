#!/usr/bin/env python3
"""B0 — always-insufficient baseline (staff manual §11.2).

A format-valid trivial system: it always returns ``insufficient`` with no
evidence. Its purpose is schema/evaluator sanity — the scientific lower bound
every team must reproduce before claiming innovation (§11.1). It uses no model,
no retrieval, and only the Python standard library.

Run:
    python -m sciaudit.baselines.b0_always_insufficient --input in.jsonl --output out.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys


def predict(instance: dict) -> dict:
    """Return the trivial always-insufficient prediction for one instance."""
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
    """Read a student-input JSONL, write one prediction per line. Returns count."""
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
    parser = argparse.ArgumentParser(description="B0 always-insufficient baseline.")
    parser.add_argument("--input", required=True, help="Path to student-input JSONL.")
    parser.add_argument("--output", required=True, help="Path to write prediction JSONL.")
    args = parser.parse_args(argv)
    n = run(args.input, args.output)
    print(f"B0 wrote {n} prediction(s) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
