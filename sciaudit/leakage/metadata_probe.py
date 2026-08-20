#!/usr/bin/env python3
"""Проба на метаданных (мануал §10.2).

Учит решающее дерево предсказывать gold-вердикт **по всему, кроме текста
evidence**: тип claim, его длину, число единиц evidence и их длины, состав
модальностей и source_kind, идентификатор статьи и форму instance_id. Если
такая модель бьёт случайный ответ значимо — датасет решается в обход задачи,
и метрики на нём меряют не аудит.

Дерево, а не «просто модель»: когда проба срабатывает, штабу нужно знать, какое
поле течёт, а не только что что-то течёт. Корневой сплит называет виновника
прямо, и отчёт его печатает.

Что делать при срабатывании (§10.2): убрать поле, перебалансировать классы или
перегенерировать идентификаторы — но **не** править формулировки claim: это
чинит симптом и ломает сам предмет измерения.

Запуск::

    python -m sciaudit.leakage.metadata_probe \\
        --inputs data_public/public_dev/inputs.jsonl \\
        --gold data_public/public_dev/gold.jsonl

Коды возврата: 0 — чисто или подозрение, 1 — утечка, 2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter

from sciaudit.leakage.probe_common import (
    exit_code,
    permutation_test,
    read_labeled,
    render_report,
)

#: Глубина дерева. Два уровня — это «пара полей», ровно та сложность, на которой
#: утечка ещё читается глазами. Глубже дерево начинает запоминать инстансы.
DEFAULT_MAX_DEPTH = 2

#: Ниже этого числа инстансов в узле сплит не делается: на трёх примерах любое
#: разделение идеально и ничего не значит.
MIN_LEAF = 3


def features_of(instance):
    """Признаки §10.2. Текста evidence и текста claim здесь нет и быть не должно."""
    claim = instance.get("claim", {})
    pack = instance.get("evidence_pack", []) or []
    lengths = [len(unit.get("text", "")) for unit in pack] or [0]
    instance_id = instance.get("instance_id", "")

    row = {
        "claim_length_chars": float(len(claim.get("text", ""))),
        "claim_length_words": float(len(claim.get("text", "").split())),
        "evidence_count": float(len(pack)),
        "evidence_len_mean": statistics.fmean(lengths),
        "evidence_len_max": float(max(lengths)),
        "evidence_len_min": float(min(lengths)),
        "evidence_len_total": float(sum(lengths)),
        "allowed_eids_count": float(len(instance.get("allowed_evidence_ids") or [])),
        "id_length": float(len(instance_id)),
        "id_digits": float(sum(character.isdigit() for character in instance_id)),
    }

    # Категориальные поля разворачиваются в индикаторы: так корневой сплит
    # называет не «признак 7», а конкретное значение конкретного поля.
    row[f"claim_type={claim.get('claim_type')}"] = 1.0
    row[f"scope={claim.get('scope')}"] = 1.0
    row[f"paper_id={instance.get('paper_id')}"] = 1.0
    for unit in pack:
        row[f"modality={unit.get('modality')}"] = row.get(
            f"modality={unit.get('modality')}", 0.0) + 1.0
        row[f"source_kind={unit.get('source_kind')}"] = row.get(
            f"source_kind={unit.get('source_kind')}", 0.0) + 1.0
    if pack:
        row[f"first_source_kind={pack[0].get('source_kind')}"] = 1.0
        row[f"last_source_kind={pack[-1].get('source_kind')}"] = 1.0
    return row


def _gini(labels):
    n = len(labels)
    return 1.0 - sum((count / n) ** 2 for count in Counter(labels).values())


def _best_split(rows, labels, names):
    """Пара «признак, порог» с наибольшим уменьшением Gini. ``None``, если нет."""
    best = None
    parent = _gini(labels)
    for name in names:
        values = sorted({row.get(name, 0.0) for row in rows})
        for left_value, right_value in zip(values, values[1:]):
            threshold = (left_value + right_value) / 2.0
            left = [label for row, label in zip(rows, labels)
                    if row.get(name, 0.0) <= threshold]
            right = [label for row, label in zip(rows, labels)
                     if row.get(name, 0.0) > threshold]
            if len(left) < MIN_LEAF or len(right) < MIN_LEAF:
                continue
            weighted = (len(left) * _gini(left) + len(right) * _gini(right)) / len(rows)
            gain = parent - weighted
            if gain > 1e-12 and (best is None or gain > best[0]):
                best = (gain, name, threshold)
    return best


def fit_tree(rows, labels, max_depth=DEFAULT_MAX_DEPTH, names=None):
    """Маленькое CART-дерево на Gini. Лист — самый частый класс в узле."""
    if names is None:
        names = sorted({name for row in rows for name in row})
    leaf = {"leaf": Counter(labels).most_common(1)[0][0]}
    if max_depth <= 0 or len(set(labels)) == 1 or len(rows) < 2 * MIN_LEAF:
        return leaf

    split = _best_split(rows, labels, names)
    if split is None:
        return leaf

    _, name, threshold = split
    left_rows, left_labels, right_rows, right_labels = [], [], [], []
    for row, label in zip(rows, labels):
        if row.get(name, 0.0) <= threshold:
            left_rows.append(row)
            left_labels.append(label)
        else:
            right_rows.append(row)
            right_labels.append(label)

    return {
        "name": name,
        "threshold": threshold,
        "left": fit_tree(left_rows, left_labels, max_depth - 1, names),
        "right": fit_tree(right_rows, right_labels, max_depth - 1, names),
    }


def predict_tree(tree, row):
    while "leaf" not in tree:
        tree = tree["left"] if row.get(tree["name"], 0.0) <= tree["threshold"] \
            else tree["right"]
    return tree["leaf"]


def describe_tree(tree, indent="  "):
    if "leaf" in tree:
        return [f"{indent}-> {tree['leaf']}"]
    head = f"{indent}{tree['name']} <= {tree['threshold']:.3g}"
    return ([head]
            + describe_tree(tree["left"], indent + "  ")
            + [f"{indent}иначе"]
            + describe_tree(tree["right"], indent + "  "))


def make_fit_predict(max_depth):
    def fit_predict(train_rows, train_labels, test_row):
        return predict_tree(fit_tree(train_rows, train_labels, max_depth), test_row)
    return fit_predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проба §10.2: предсказуем ли вердикт по одним метаданным.")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        pairs = read_labeled(args.inputs, args.gold)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows = [features_of(instance) for instance, _ in pairs]
    labels = [verdict for _, verdict in pairs]

    fit_predict = make_fit_predict(args.max_depth)
    observed, p_value, sampled = permutation_test(
        fit_predict, rows, labels, permutations=args.permutations, seed=args.seed)

    tree = fit_tree(rows, labels, args.max_depth)
    extra = ["дерево на всех данных (что именно оно нашло):"]
    extra += [line[2:] if line.startswith("  ") else line
              for line in describe_tree(tree)]

    verdict, report = render_report("metadata_probe", observed, p_value, labels,
                                    sampled, extra)
    print(report)
    return exit_code(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
