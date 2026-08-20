#!/usr/bin/env python3
"""B1 — BM25-ретривал плюс структурированный аудит моделью (мануал §11.3).

Первый нетривиальный бейзлайн после B0. Из evidence pack отбираются top-k
единиц по BM25, из них строится строгий JSON-промпт, ответ модели
нормализуется до объекта предсказания.

Три инварианта, которые модуль обязан держать независимо от того, что вернула
модель:

* ``predicted_eids`` — подмножество ``allowed_evidence_ids`` (мануал §5.2);
* ``issue_tags`` — только из словаря схемы предсказания (Listing 3);
* невалидный ответ модели не роняет прогон, а даёт безопасный отказ.

Последнее опасно тем, что молчаливый отказ неотличим от осторожной системы, а
evaluator считает selective risk и coverage. Поэтому каждый уход в fallback
печатается в stderr, а прогон, где в fallback ушли все инстансы, завершается
ненулевым кодом: это ошибка конфигурации, а не результат измерения.

Запуск::

    python -m sciaudit.baselines.b1_bm25_llm \\
        --input inputs.jsonl --output predictions.jsonl \\
        --model-command "your-open-model-command"

Коды возврата: 0 — успех, 1 — все инстансы ушли в fallback, 2 — ошибка
использования или чтения входа.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

from sciaudit.schemas import read_jsonl

VERDICTS = frozenset({"warranted", "overclaimed", "contradicted", "insufficient"})
SAFE_FALLBACK_VERDICT = "insufficient"

#: Словарь issue-тегов из ``schemas/prediction.schema.json``. Продублирован
#: здесь намеренно: модуль не должен читать файл схемы в рантайме. Расхождение
#: сторожит тест ``test_issue_tags_match_the_prediction_schema``.
ISSUE_TAGS = frozenset({
    "numerical_inconsistency",
    "claim_stronger_than_evidence",
    "unsupported_generalization",
    "missing_ablation_support",
    "non_comparable_baseline",
    "weak_statistical_support",
    "evidence_missing_or_incomplete",
    "caption_chart_mismatch",
})

#: Текст обоснования безопасного отказа. Единственный источник истины: по нему
#: ``main`` считает долю fallback, поэтому строку нельзя менять по месту.
FALLBACK_RATIONALE = "Model output could not be parsed after retries; safe fallback used."

DEFAULT_MODEL_NAME = "b1-open-model"
MOCK_MODEL_NAME = "mocked_model"
DEFAULT_TIMEOUT_SECONDS = 60  # согласовано с configs/models.yaml


class ModelCallError(RuntimeError):
    """Модель не ответила пригодным для разбора текстом."""


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def bm25_rank(query, evidence_pack, top_k=3, k1=1.5, b=0.75):
    """Top-k единиц evidence по BM25. Сортировка стабильная, порядок детерминирован."""
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


def build_prompt(instance, retrieved_evidence):
    claim = instance.get("claim", {})
    claim_text = claim.get("text") or claim.get("normalized_claim") or ""

    evidence_text = "\n".join(
        f"- {ev.get('eid')}: {ev.get('text', '')}"
        for ev in retrieved_evidence
    )

    allowed_tags = ", ".join(sorted(ISSUE_TAGS))

    return f"""You are auditing whether a claim is supported by the supplied evidence only.

Return strict JSON only with these fields:
- verdict: one of warranted, overclaimed, contradicted, insufficient
- confidence: number from 0 to 1
- predicted_eids: list of evidence IDs, drawn only from the evidence below
- issue_tags: list, each item one of: {allowed_tags}
- abstain: boolean
- rationale_short: short explanation

Claim:
{claim_text}

Evidence:
{evidence_text}
"""


def call_model(prompt, model_command=None, timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    """Выполнить команду модели и вернуть stdout.

    Транспорт — произвольная shell-команда. Это временное решение: по issue #12
    доступ к модели должен идти через OpenAI-совместимый профиль из
    ``configs/models.yaml`` с ключом из переменной окружения. Пока клиента нет,
    команда передаётся как есть, и ключ в ней виден в ``ps`` и в history.
    """
    if not model_command:
        raise ModelCallError("model command is not configured")

    try:
        completed = subprocess.run(
            model_command,
            input=prompt,
            text=True,
            capture_output=True,
            shell=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelCallError(f"model command timed out after {timeout_seconds}s") from exc
    except OSError as exc:
        raise ModelCallError(f"model command could not be started: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "no stderr"
        raise ModelCallError(f"model command exited with {completed.returncode}: {tail}")

    return completed.stdout.strip()


def parse_model_json(text):
    if not text or not text.strip():
        raise ModelCallError("model returned empty output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ModelCallError("no JSON object found in model output")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ModelCallError(f"embedded JSON object is malformed: {exc}") from exc


def _envelope(instance_id, verdict, confidence, predicted_eids, issue_tags, abstain,
              rationale, model_name, runtime_seconds):
    return {
        "instance_id": instance_id,
        "verdict": verdict,
        "confidence": confidence,
        "predicted_eids": predicted_eids,
        "issue_tags": issue_tags,
        "abstain": abstain,
        "rationale_short": rationale,
        "runtime_seconds": runtime_seconds,
        "estimated_cost": {
            "gpu_seconds": 0.0,
            "api_cost_usd": 0.0,
        },
        "system_info": {
            "model": model_name,
            "uses_numeric_checker": False,
            "uses_lora": False,
            "uses_vlm": False,
        },
    }


def safe_prediction(instance_id, model_name, runtime_seconds,
                    rationale=FALLBACK_RATIONALE):
    """Безопасный отказ: формат валиден, вердикт консервативен, evidence не назван."""
    return _envelope(instance_id, SAFE_FALLBACK_VERDICT, 0.0, [], [], True,
                     rationale, model_name, runtime_seconds)


def normalize_prediction(raw, instance_id, allowed_eids, model_name, runtime_seconds):
    """Привести ответ модели к объекту схемы предсказания.

    Всё, что схема не примет, отбрасывается здесь: чужие вердикты, теги вне
    словаря, eid вне ``allowed_evidence_ids``, confidence за пределами [0, 1].
    """
    verdict = raw.get("verdict", SAFE_FALLBACK_VERDICT)
    if verdict not in VERDICTS:
        verdict = SAFE_FALLBACK_VERDICT

    confidence = raw.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    predicted_eids = raw.get("predicted_eids", [])
    if not isinstance(predicted_eids, list):
        predicted_eids = []
    seen = set()
    predicted_eids = [
        eid for eid in predicted_eids
        if isinstance(eid, str) and eid in allowed_eids
        and not (eid in seen or seen.add(eid))
    ]

    issue_tags = raw.get("issue_tags", [])
    if not isinstance(issue_tags, list):
        issue_tags = []
    seen_tags = set()
    issue_tags = [
        tag for tag in issue_tags
        if isinstance(tag, str) and tag in ISSUE_TAGS
        and not (tag in seen_tags or seen_tags.add(tag))
    ]

    abstain = raw.get("abstain", False)
    if not isinstance(abstain, bool):
        abstain = False

    rationale_short = raw.get("rationale_short", "")
    if not isinstance(rationale_short, str) or not rationale_short.strip():
        rationale_short = "B1 produced a normalized structured prediction."

    return _envelope(instance_id, verdict, confidence, predicted_eids, issue_tags,
                     abstain, rationale_short[:500], model_name, runtime_seconds)


def audit_instance(instance, top_k=3, model_command=None, model_name=None, retries=1,
                   model_fn=None, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, log=None):
    """Одно предсказание. Возвращает ``(prediction, fallback_reason | None)``."""
    start = time.perf_counter()

    if "instance_id" not in instance:
        raise ValueError("input instance has no instance_id")
    instance_id = instance["instance_id"]

    claim = instance.get("claim", {})
    claim_text = claim.get("text") or claim.get("normalized_claim") or ""
    evidence_pack = instance.get("evidence_pack", [])
    allowed_eids = set(instance.get("allowed_evidence_ids") or [])
    if not allowed_eids:
        allowed_eids = {ev.get("eid") for ev in evidence_pack if ev.get("eid")}

    # Идентификатор модели декларируется вызывающим и никогда не выводится ни из
    # команды запуска (в ней может быть ключ), ни из имени python-функции.
    resolved_model = model_name or (MOCK_MODEL_NAME if model_fn else DEFAULT_MODEL_NAME)

    retrieved = bm25_rank(claim_text, evidence_pack, top_k=top_k)
    prompt = build_prompt(instance, retrieved)

    last_reason = "model was never called"
    for attempt in range(1, retries + 2):
        try:
            if model_fn is not None:
                model_output = model_fn(prompt)
                if not isinstance(model_output, str):
                    raise ModelCallError("model_fn did not return a string")
            else:
                model_output = call_model(prompt, model_command=model_command,
                                          timeout_seconds=timeout_seconds)

            raw = parse_model_json(model_output)
            if not isinstance(raw, dict):
                raise ModelCallError("model JSON is not an object")

            return normalize_prediction(
                raw=raw,
                instance_id=instance_id,
                allowed_eids=allowed_eids,
                model_name=resolved_model,
                runtime_seconds=time.perf_counter() - start,
            ), None
        except ModelCallError as exc:
            last_reason = str(exc)
            if log is not None:
                log(f"B1: {instance_id}: попытка {attempt}/{retries + 1} — {last_reason}")

    return safe_prediction(
        instance_id=instance_id,
        model_name=resolved_model,
        runtime_seconds=time.perf_counter() - start,
    ), last_reason


def run(input_path, output_path, top_k=3, model_command=None,
        model_name=None, retries=1, model_fn=None,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS, log=None):
    """Прочитать входы, записать по одному предсказанию на строку, вернуть их список."""
    if model_command is None and model_fn is None:
        raise ValueError("B1 requires either --model-command or model_fn.")
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if retries < 0:
        raise ValueError("--retries must not be negative.")

    instances = [obj for _, obj in read_jsonl(input_path)]

    predictions = []
    for instance in instances:
        prediction, _ = audit_instance(
            instance,
            top_k=top_k,
            model_command=model_command,
            model_name=model_name,
            retries=retries,
            model_fn=model_fn,
            timeout_seconds=timeout_seconds,
            log=log,
        )
        predictions.append(prediction)

    write_jsonl(output_path, predictions)
    return predictions


def count_fallbacks(predictions):
    """Сколько предсказаний — безопасный отказ, а не ответ модели."""
    return sum(1 for p in predictions if p.get("rationale_short") == FALLBACK_RATIONALE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Бейзлайн B1: BM25-ретривал + структурированный аудит моделью.")
    parser.add_argument("--input", required=True, help="Путь к JSONL со входами.")
    parser.add_argument("--output", required=True, help="Путь к JSONL с предсказаниями.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Сколько единиц evidence отбирает BM25.")
    parser.add_argument("--model-command", default=None,
                        help="Shell-команда инференса. Промпт подаётся в stdin.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Идентификатор модели для system_info.model. "
                             "Именно модель, не команда запуска и не строка с ключом.")
    parser.add_argument("--retries", type=int, default=1,
                        help="Сколько раз повторить запрос после неразобранного ответа.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Таймаут одного вызова модели.")
    args = parser.parse_args(argv)

    def log(message):
        print(message, file=sys.stderr)

    try:
        predictions = run(
            input_path=args.input,
            output_path=args.output,
            top_k=args.top_k,
            model_command=args.model_command,
            model_name=args.model_name,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            log=log,
        )
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    total = len(predictions)
    fallbacks = count_fallbacks(predictions)
    print(f"B1: записано предсказаний — {total}, файл {args.output}; "
          f"безопасных отказов — {fallbacks}", file=sys.stderr)

    if total and fallbacks == total:
        print("ERROR: ни один инстанс не получил разобранного ответа модели. "
              "Это отказ конфигурации, а не осторожная система — метрики по такому "
              "прогону считать нельзя.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
