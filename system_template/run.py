#!/usr/bin/env python3
"""SciAudit-Stress system template.

Scaffold placeholder: reads a student-input JSONL and emits a SCHEMA-VALID
prediction JSONL (one prediction per input). It does NO real auditing yet — its
job is to prove the rail runs end-to-end ("rails before trains", staff manual
§1.3, §8.1). Teams replace `audit_instance` with their actual pipeline; the
external contract (--input / --output) must stay identical.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def audit_instance(instance: dict) -> dict:
    """Return a schema-valid prediction for one student-input instance.

    Placeholder policy: always "insufficient" with no evidence — mirrors the B0
    sanity baseline (staff manual §11.2). Replace with a real system.
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
    parser = argparse.ArgumentParser(description="SciAudit-Stress system runner.")
    parser.add_argument("--input", required=True, help="Path to student-input JSONL.")
    parser.add_argument("--output", required=True, help="Path to write prediction JSONL.")
    args = parser.parse_args(argv)

    n = run(args.input, args.output)
    print(f"Wrote {n} prediction(s) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
