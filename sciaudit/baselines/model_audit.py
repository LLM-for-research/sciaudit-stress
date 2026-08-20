#!/usr/bin/env python3
"""Общий слой вызова модели для бейзлайнов B1–B4 (мануал §11.3–§11.4).

Вынесено сюда всё, что у бейзлайнов на модели одинаково: построение промпта,
вызов, разбор ответа, нормализация до объекта схемы предсказания, безопасный
отказ и контракт CLI. Различаются бейзлайны ровно одним — тем, какие единицы
evidence попадают в промпт. Эта разница выражена функцией-селектором:

* B1 отдаёт top-k по BM25;
* B2 отдаёт весь evidence pack.

Тогда сравнение B1 против B2 измеряет вклад ретрива, а не разницу в промптах,
парсинге или обработке отказов — иначе научный контроль ничего не контролирует.

Три инварианта держатся здесь и потому одинаковы для всех бейзлайнов:

* ``predicted_eids`` — подмножество ``allowed_evidence_ids`` (§5.2);
* ``issue_tags`` — только из словаря схемы предсказания (Listing 3);
* невалидный ответ модели даёт безопасный отказ, а не падение прогона.

Молчаливый отказ опасен: evaluator считает selective risk и coverage, и мёртвая
модель без предупреждения выглядит как образцово осторожная система. Поэтому
каждый уход в отказ печатается в stderr, а прогон, где отказались все инстансы,
завершается кодом 1.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

from sciaudit.schemas import read_jsonl

VERDICTS = frozenset({"warranted", "overclaimed", "contradicted", "insufficient"})
SAFE_FALLBACK_VERDICT = "insufficient"

#: Словарь issue-тегов из ``schemas/prediction.schema.json``. Продублирован
#: намеренно: модуль не читает файл схемы в рантайме. Расхождение сторожит тест
#: ``test_issue_tags_match_the_prediction_schema``.
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
#: считается доля отказов, поэтому строку нельзя менять по месту.
FALLBACK_RATIONALE = "Model output could not be parsed after retries; safe fallback used."

DEFAULT_MODEL_NAME = "open-model"
MOCK_MODEL_NAME = "mocked_model"
DEFAULT_TIMEOUT_SECONDS = 60  # согласовано с configs/models.yaml


class ModelCallError(RuntimeError):
    """Модель не ответила пригодным для разбора текстом."""


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def claim_text_of(instance):
    claim = instance.get("claim", {})
    return claim.get("text") or claim.get("normalized_claim") or ""


def build_prompt(claim_text, evidence_units):
    evidence_text = "\n".join(
        f"- {ev.get('eid')}: {ev.get('text', '')}" for ev in evidence_units
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

    Транспорт — произвольная shell-команда. Временное решение: по issue #12
    доступ должен идти через OpenAI-совместимый профиль из ``configs/models.yaml``
    с ключом из переменной окружения. Когда клиент появится, подменяется ровно
    эта функция — остальной слой не меняется.
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
              rationale, model_name, runtime_seconds, uses_numeric_checker=False):
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
            "uses_numeric_checker": uses_numeric_checker,
            "uses_lora": False,
            "uses_vlm": False,
        },
    }


def safe_prediction(instance_id, model_name, runtime_seconds,
                    rationale=FALLBACK_RATIONALE):
    """Безопасный отказ: формат валиден, вердикт консервативен, evidence не назван."""
    return _envelope(instance_id, SAFE_FALLBACK_VERDICT, 0.0, [], [], True,
                     rationale, model_name, runtime_seconds)


def _dedupe_allowed(values, allowed):
    seen = set()
    out = []
    for value in values:
        if isinstance(value, str) and value in allowed and value not in seen:
            seen.add(value)
            out.append(value)
    return out


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
    predicted_eids = _dedupe_allowed(predicted_eids, allowed_eids)

    issue_tags = raw.get("issue_tags", [])
    if not isinstance(issue_tags, list):
        issue_tags = []
    issue_tags = _dedupe_allowed(issue_tags, ISSUE_TAGS)

    abstain = raw.get("abstain", False)
    if not isinstance(abstain, bool):
        abstain = False

    rationale_short = raw.get("rationale_short", "")
    if not isinstance(rationale_short, str) or not rationale_short.strip():
        rationale_short = "Normalized structured prediction."

    return _envelope(instance_id, verdict, confidence, predicted_eids, issue_tags,
                     abstain, rationale_short[:500], model_name, runtime_seconds)


def audit_instance(instance, select_evidence, model_command=None, model_name=None,
                   retries=1, model_fn=None, timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                   log=None):
    """Одно предсказание. Возвращает ``(prediction, fallback_reason | None)``.

    ``select_evidence(claim_text, evidence_pack)`` — единственное место, которым
    бейзлайны на модели отличаются друг от друга.
    """
    start = time.perf_counter()

    if "instance_id" not in instance:
        raise ValueError("input instance has no instance_id")
    instance_id = instance["instance_id"]

    claim_text = claim_text_of(instance)
    evidence_pack = instance.get("evidence_pack", [])
    allowed_eids = set(instance.get("allowed_evidence_ids") or [])
    if not allowed_eids:
        allowed_eids = {ev.get("eid") for ev in evidence_pack if ev.get("eid")}

    # Идентификатор модели декларируется вызывающим и никогда не выводится ни из
    # команды запуска (в ней может быть ключ), ни из имени python-функции.
    resolved_model = model_name or (MOCK_MODEL_NAME if model_fn else DEFAULT_MODEL_NAME)

    prompt = build_prompt(claim_text, select_evidence(claim_text, evidence_pack))

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
                log(f"{instance_id}: попытка {attempt}/{retries + 1} — {last_reason}")

    return safe_prediction(
        instance_id=instance_id,
        model_name=resolved_model,
        runtime_seconds=time.perf_counter() - start,
    ), last_reason


def run(input_path, output_path, select_evidence, model_command=None, model_name=None,
        retries=1, model_fn=None, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, log=None):
    """Прочитать входы, записать по предсказанию на строку, вернуть их список."""
    if model_command is None and model_fn is None:
        raise ValueError("this baseline requires either --model-command or model_fn.")
    if retries < 0:
        raise ValueError("--retries must not be negative.")

    predictions = []
    for _, instance in read_jsonl(input_path):
        prediction, _ = audit_instance(
            instance,
            select_evidence,
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


def build_arg_parser(description, default_model_name=DEFAULT_MODEL_NAME):
    """Общий контракт CLI. Бейзлайн вправе добавить свои аргументы поверх."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Путь к JSONL со входами.")
    parser.add_argument("--output", required=True, help="Путь к JSONL с предсказаниями.")
    parser.add_argument("--model-command", default=None,
                        help="Shell-команда инференса. Промпт подаётся в stdin.")
    parser.add_argument("--model-name", default=default_model_name,
                        help="Идентификатор модели для system_info.model. "
                             "Именно модель, не команда запуска и не строка с ключом.")
    parser.add_argument("--retries", type=int, default=1,
                        help="Сколько раз повторить запрос после неразобранного ответа.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Таймаут одного вызова модели.")
    return parser


def run_cli(args, select_evidence, label):
    """Общее тело ``main`` бейзлайна. Возвращает код возврата процесса."""
    def log(message):
        print(f"{label}: {message}", file=sys.stderr)

    try:
        predictions = run(
            input_path=args.input,
            output_path=args.output,
            select_evidence=select_evidence,
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
    print(f"{label}: записано предсказаний — {total}, файл {args.output}; "
          f"безопасных отказов — {fallbacks}", file=sys.stderr)

    if total and fallbacks == total:
        print("ERROR: ни один инстанс не получил разобранного ответа модели. "
              "Это отказ конфигурации, а не осторожная система — метрики по такому "
              "прогону считать нельзя.", file=sys.stderr)
        return 1

    return 0
