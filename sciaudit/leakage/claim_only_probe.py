#!/usr/bin/env python3
"""Проба на одном тексте claim (мануал §10.3).

Учит наивный байесовский классификатор предсказывать gold-вердикт **по словам
claim и только по ним** — evidence не показывается вовсе. Если такая модель
работает заметно лучше случая, метку выдаёт формулировка: claim со словами
«all», «always», «consistently», «state-of-the-art» тянут за собой
``overclaimed`` независимо от того, что лежит в паке.

**Чинить это переписыванием формулировок нельзя** — так прямо сказано в §10.3, и
это важнее самой пробы. Сильные формулировки должны встречаться и среди
``warranted``, а осторожные — среди ``insufficient``; лечение — балансировка
случаев, а не цензура языка. Иначе бенчмарк начнёт измерять вежливость
авторов вместо достаточности доказательств.

Наивный Байес, а не что-то сильнее: проба обязана быть слабой. Мощная модель
найдёт связь в любом тексте, и гейт превратится в генератор ложных тревог. Здесь
важно поймать грубую утечку — «одно слово решает», — а не тонкую.

Отчёт печатает слова с наибольшим перекосом по классам: если проба сработала,
это и есть список того, что надо балансировать.

Запуск::

    python -m sciaudit.leakage.claim_only_probe \\
        --inputs data_public/public_dev/inputs.jsonl \\
        --gold data_public/public_dev/gold.jsonl

Коды возврата: 0 — чисто или подозрение, 1 — утечка, 2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict

from sciaudit.leakage.probe_common import (
    exit_code,
    permutation_test,
    read_labeled,
    render_report,
)

_TOKEN = re.compile(r"[a-zа-яё0-9]+")

#: Сглаживание Лапласа. Единица — стандартный выбор; на десятках инстансов
#: меньшее значение делает редкое слово решающим.
ALPHA = 1.0

#: Сколько слов-индикаторов печатать на класс.
TOP_TOKENS = 5


def tokenize(text):
    return _TOKEN.findall((text or "").lower())


def claim_tokens(instance):
    """Только текст claim. Ни evidence, ни типа, ни длины пака."""
    return tokenize(instance.get("claim", {}).get("text", ""))


def fit_naive_bayes(documents, labels, alpha=ALPHA):
    priors = Counter(labels)
    counts = defaultdict(Counter)
    totals = Counter()
    vocabulary = set()
    for tokens, label in zip(documents, labels):
        for token in tokens:
            counts[label][token] += 1
            totals[label] += 1
            vocabulary.add(token)
    return {"priors": priors, "counts": counts, "totals": totals,
            "vocabulary": vocabulary, "alpha": alpha, "n": len(labels)}


def predict_naive_bayes(model, tokens):
    size = len(model["vocabulary"]) or 1
    best_label, best_score = None, None
    for label, prior in sorted(model["priors"].items()):
        score = math.log(prior / model["n"])
        denominator = model["totals"][label] + model["alpha"] * size
        for token in tokens:
            if token not in model["vocabulary"]:
                continue
            score += math.log((model["counts"][label][token] + model["alpha"])
                              / denominator)
        if best_score is None or score > best_score:
            best_label, best_score = label, score
    return best_label


def fit_predict(train_documents, train_labels, test_document):
    return predict_naive_bayes(fit_naive_bayes(train_documents, train_labels),
                               test_document)


def indicative_tokens(documents, labels, top=TOP_TOKENS, min_count=2):
    """Слова, чья доля в классе сильнее всего превышает долю в остальных.

    Это и есть список к балансировке, если проба сработала: не «claim слишком
    сильный», а «вот эти слова встречаются только в одном классе».
    """
    per_label = defaultdict(Counter)
    overall = Counter()
    for tokens, label in zip(documents, labels):
        for token in set(tokens):
            per_label[label][token] += 1
            overall[token] += 1

    lines = []
    for label in sorted(per_label):
        in_label = sum(1 for value in labels if value == label)
        scored = []
        for token, count in per_label[label].items():
            if overall[token] < min_count:
                continue
            share_in = count / in_label
            share_out = (overall[token] - count) / max(1, len(labels) - in_label)
            scored.append((share_in - share_out, token, count, overall[token]))
        scored.sort(reverse=True)
        if scored:
            words = ", ".join(f"{token} ({count}/{total})"
                              for _, token, count, total in scored[:top])
            lines.append(f"{label}: {words}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проба §10.3: предсказуем ли вердикт по одному тексту claim.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        pairs = read_labeled(args.inputs, args.gold)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    documents = [claim_tokens(instance) for instance, _ in pairs]
    labels = [verdict for _, verdict in pairs]

    observed, p_value, sampled = permutation_test(
        fit_predict, documents, labels,
        permutations=args.permutations, seed=args.seed)

    extra = ["слова с наибольшим перекосом (встречаемость в классе / всего):"]
    extra += indicative_tokens(documents, labels)

    verdict, report = render_report("claim_only_probe", observed, p_value, labels,
                                    sampled, extra)
    print(report)
    return exit_code(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
