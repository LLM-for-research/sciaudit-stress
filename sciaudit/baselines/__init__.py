"""Baselines B0-B4 (Task 4 — implemented in later issues).

Each baseline is a module runnable as:
    python -m sciaudit.baselines.b0_always_insufficient --input <in> --output <out>

Planned (staff manual §11):
- b0_always_insufficient   : schema/evaluator sanity
- b1_bm25_llm              : BM25 retrieval + structured open-model audit
- b2_fullpack_llm          : full evidence-pack structured audit
- b3_numeric_checker_llm   : deterministic numeric/table checker + LLM
- b4_selective_threshold   : selective abstention baseline
"""
