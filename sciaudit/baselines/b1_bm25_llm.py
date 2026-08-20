#!/usr/bin/env python3
"""B1 — BM25-ретривал плюс структурированный аудит моделью (мануал §11.3).

Первый нетривиальный бейзлайн после B0. Из evidence pack отбираются top-k
единиц по BM25, из них строится строгий JSON-промпт, ответ модели
нормализуется до объекта предсказания.

Всё, что у бейзлайнов на модели одинаково — промпт, вызов, разбор,
нормализация, безопасный отказ, контракт CLI — живёт в
:mod:`sciaudit.baselines.model_audit`. Здесь остаётся только ретрив: ровно им
B1 отличается от контрольного B2, который получает весь пак целиком.

Запуск::

    python -m sciaudit.baselines.b1_bm25_llm \\
        --input inputs.jsonl --output predictions.jsonl \\
        --top-k 3 --model-command "your-open-model-command"

Коды возврата: 0 — успех, 1 — все инстансы ушли в безопасный отказ,
2 — ошибка использования или чтения входа.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict

from sciaudit.baselines.model_audit import (  # noqa: F401  (публичная поверхность модуля)
    DEFAULT_TIMEOUT_SECONDS,
    FALLBACK_RATIONALE,
    ISSUE_TAGS,
    MOCK_MODEL_NAME,
    SAFE_FALLBACK_VERDICT,
    VERDICTS,
    ModelCallError,
    build_arg_parser,
    build_prompt,
    call_model,
    count_fallbacks,
    normalize_prediction,
    parse_model_json,
    run_cli,
    safe_prediction,
    write_jsonl,
)
from sciaudit.baselines import model_audit

LABEL = "B1"
DEFAULT_TOP_K = 3
#: Заглушка для system_info.model, если идентификатор не объявлен явно.
DEFAULT_MODEL_NAME = "b1-open-model"


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def bm25_rank(query, evidence_pack, top_k=DEFAULT_TOP_K, k1=1.5, b=0.75):
    """Top-k единиц evidence по BM25. Порядок детерминирован при равных счётах."""
    if not evidence_pack:
        return []

    docs = [tokenize(ev.get("text", "")) for ev in evidence_pack]
    query_terms = tokenize(query)

    doc_lens = [len(doc) for doc in docs]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0

    df = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = []
    n_docs = len(docs)

    for idx, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue

            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_lens[idx] / avgdl) if avgdl else 1.0
            score += idf * (tf[term] * (k1 + 1)) / denom

        scores.append((score, idx))

    scores.sort(key=lambda pair: (-pair[0], pair[1]))
    return [evidence_pack[idx] for _, idx in scores[:top_k]]


def make_selector(top_k=DEFAULT_TOP_K):
    """Селектор B1: top-k по BM25 относительно текста claim."""
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    def select(claim_text, evidence_pack):
        return bm25_rank(claim_text, evidence_pack, top_k=top_k)

    return select


def run(input_path, output_path, top_k=DEFAULT_TOP_K, **kwargs):
    """Прогнать B1. ``top_k`` — единственный аргумент сверх общего контракта."""
    return model_audit.run(input_path, output_path, make_selector(top_k), **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        "Бейзлайн B1: BM25-ретривал + структурированный аудит моделью.",
        default_model_name=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Сколько единиц evidence отбирает BM25.")
    args = parser.parse_args(argv)

    try:
        selector = make_selector(args.top_k)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return run_cli(args, selector, LABEL)


if __name__ == "__main__":
    raise SystemExit(main())
