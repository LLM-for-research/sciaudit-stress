#!/usr/bin/env python3
"""Общая часть проб на утечку метки (мануал §10.2 и §10.3).

Обе пробы устроены одинаково: обучить простую модель на признаках, которых
быть достаточно **не должно**, и проверить, не предсказывает ли она gold-вердикт
лучше, чем следует из одного лишь распределения классов. Если предсказывает —
датасет решается в обход evidence, и метрики на нём меряют не аудит.

Три решения, общие для обеих проб.

**Leave-one-out вместо k-fold.** Инстансов десятки, классов четыре; при пяти
фолдах в тестовой части регулярно оказывается класс, которого не было в
обучающей, и точность начинает зависеть от того, как легли фолды. LOO этой
зависимости не имеет и при таких размерах ничего не стоит.

**Перестановочный тест вместо порога «заметного превосходства».** Мануал
говорит «beats baselines by a meaningful margin», но что такое meaningful при
n = 24, из формулировки не следует: разрыв в 8 процентных пунктов — это два
инстанса. Поэтому метки многократно перемешиваются, проба переобучается на
каждой перестановке, и наблюдённая точность сравнивается с распределением
случайных. p-value отвечает на правильный вопрос: как часто такой результат
получается, когда связи нет вовсе.

**Гейт различает утечку и подозрение.** Уверенная утечка валит сборку (код 1),
подозрение печатается и пропускает: на маленьком срезе граница между «связь
есть» и «повезло» широкая, и превращать её в бинарную — значит либо пугать
ложными тревогами, либо пропускать настоящие.
"""
from __future__ import annotations

import json
import random
from collections import Counter

#: Ниже этого p-value проба считается сработавшей: связь метки с признаками
#: есть, датасет заражён.
ALPHA_LEAK = 0.05

#: Между ALPHA_LEAK и этим значением печатается подозрение без падения гейта.
ALPHA_SUSPECT = 0.20

VERDICT_CLEAN, VERDICT_SUSPECT, VERDICT_LEAK = "clean", "suspect", "leak"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_labeled(inputs_path, gold_path):
    """Сопоставить входы с gold-вердиктами по ``instance_id``."""
    inputs = read_jsonl(inputs_path)
    gold = {row["instance_id"]: row["gold"]["verdict"] for row in read_jsonl(gold_path)}

    pairs, missing = [], []
    for instance in inputs:
        verdict = gold.get(instance["instance_id"])
        if verdict is None:
            missing.append(instance["instance_id"])
        else:
            pairs.append((instance, verdict))
    if missing:
        raise ValueError("нет gold-вердикта для инстансов: " + ", ".join(missing[:5]))
    if not pairs:
        raise ValueError("во входе нет ни одного инстанса")
    return pairs


def majority_accuracy(labels):
    """Точность стратегии «всегда самый частый класс»."""
    return Counter(labels).most_common(1)[0][1] / len(labels)


def stratified_accuracy(labels):
    """Ожидаемая точность случайного ответа по частотам классов — сумма p²."""
    n = len(labels)
    return sum((count / n) ** 2 for count in Counter(labels).values())


def leave_one_out_accuracy(fit_predict, rows, labels):
    """Доля верных предсказаний, когда каждый инстанс по очереди — тест.

    ``fit_predict(train_rows, train_labels, test_row)`` возвращает метку.
    """
    correct = 0
    for i in range(len(rows)):
        train_rows = rows[:i] + rows[i + 1:]
        train_labels = labels[:i] + labels[i + 1:]
        correct += fit_predict(train_rows, train_labels, rows[i]) == labels[i]
    return correct / len(rows)


def permutation_test(fit_predict, rows, labels, permutations=500, seed=7):
    """Насколько наблюдённая точность необычна, если связи метки с признаками нет.

    Возвращает ``(наблюдённая точность, p-value, точности перестановок)``.
    Метки перемешиваются, проба переобучается целиком — включая LOO, — поэтому
    сравнивается не число с числом, а процедура с той же процедурой на шуме.
    """
    observed = leave_one_out_accuracy(fit_predict, rows, labels)
    rng = random.Random(seed)
    shuffled = list(labels)
    at_least_as_good = 0
    sampled = []
    for _ in range(permutations):
        rng.shuffle(shuffled)
        accuracy = leave_one_out_accuracy(fit_predict, rows, list(shuffled))
        sampled.append(accuracy)
        at_least_as_good += accuracy >= observed
    p_value = (1 + at_least_as_good) / (1 + permutations)
    return observed, p_value, sampled


def verdict_of(p_value):
    if p_value < ALPHA_LEAK:
        return VERDICT_LEAK
    if p_value < ALPHA_SUSPECT:
        return VERDICT_SUSPECT
    return VERDICT_CLEAN


def render_report(label, observed, p_value, labels, sampled, extra_lines=()):
    """Отчёт пробы. Всегда печатает, на чём меряли, — иначе число нечитаемо."""
    verdict = verdict_of(p_value)
    headline = {
        VERDICT_LEAK: "УТЕЧКА: метка предсказывается по признакам, которых для неё мало",
        VERDICT_SUSPECT: "ПОДОЗРЕНИЕ: связь возможна, но на этом объёме не доказана",
        VERDICT_CLEAN: "OK: связи метки с этими признаками не видно",
    }[verdict]

    lines = [
        f"{label}: {headline}",
        f"  инстансов: {len(labels)}, классов: {len(set(labels))}",
        f"  точность пробы (leave-one-out): {observed:.3f}",
        f"  всегда самый частый класс:      {majority_accuracy(labels):.3f}",
        f"  случайный ответ по частотам:    {stratified_accuracy(labels):.3f}",
        f"  перестановки: медиана {sorted(sampled)[len(sampled) // 2]:.3f}, "
        f"максимум {max(sampled):.3f}",
        f"  p-value: {p_value:.4f} (порог утечки {ALPHA_LEAK}, "
        f"подозрения {ALPHA_SUSPECT})",
    ]
    lines.extend(f"  {line}" for line in extra_lines)
    return verdict, "\n".join(lines)


def exit_code(verdict):
    return 1 if verdict == VERDICT_LEAK else 0
