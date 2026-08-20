#!/usr/bin/env python3
"""Проверка идентификаторов инстансов (мануал §5.5).

Мануал требует случайные несемантические ID и прямо запрещает такие, как
``sas_seed_00045``, ``sas_scope_stress_001``, ``goldhidden_022``,
``P017_scope_expansion``. Причина не в аккуратности: семантический ID — это
приватное поле, вынесенное в публичное имя. Увидев ``scope_stress``, система
угадает ``overclaimed``, ни разу не заглянув в evidence, и весь контроль утечек
из §10 обойдётся сбоку.

Проверяется четыре вещи:

* **форма** — ``sas_`` плюс восемь строчных букв или цифр;
* **уникальность** — один ID не встречается дважды;
* **отсутствие смысла** — в ID нет ни названий стресс-трансформаций, ни слов
  вроде ``seed``, ``gold``, ``hidden``, ни счётчиков вида ``_001``;
* **отсутствие порядка** — если отсортировать инстансы по ID, метки не должны
  сбиваться в кучи. Это ловит случай, когда ID выданы подряд по мере
  разметки: формально они случайны, а фактически соседние ID означают
  соседние по времени, то есть однотипные, инстансы.

Последняя проверка — перестановочная: считается, сколько соседних пар в
отсортированном по ID порядке имеют одинаковый вердикт, и это число
сравнивается с тем же счётчиком на случайных перестановках меток.

Запуск::

    python -m sciaudit.leakage.id_randomness_check \\
        --inputs data_public/public_dev/inputs.jsonl \\
        --gold data_public/public_dev/gold.jsonl

``--gold`` необязателен: без него проверяются форма, уникальность и смысл, но
не порядок.

Коды возврата: 0 — чисто, 1 — нарушение, 2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter

from sciaudit.leakage.probe_common import read_jsonl

#: Форма и смысл ID — проверки детерминированные: нарушение либо есть, либо нет,
#: и оно валит гейт. Порядок меток — статистика на десятках инстансов, где
#: разница между «связь» и «повезло» узкая, поэтому у неё два порога: уверенное
#: нарушение валит сборку, подозрение печатается и пропускает. Гейт, который
#: краснеет раз в пятьдесят коммитов на шуме, перестают читать.
ALPHA_ORDER_FAIL = 0.01
ALPHA_ORDER_WARN = 0.05

#: Форма из §5.5: sas_ плюс восемь строчных букв или цифр.
ID_PATTERN = re.compile(r"^sas_[a-z0-9]{8}$")

#: Куски, которые делают ID семантическим. Список из §5.4 и §5.5 плюс имена
#: скрытых срезов: увидев их в публичном ID, участник читает приватное поле.
FORBIDDEN_PARTS = (
    "seed", "stress", "scope", "expansion", "removal", "evidence",
    "numeric", "perturbation", "distractor", "gold", "hidden", "challenge",
    "autostress", "warrant", "overclaim", "contradict", "insufficient",
)

#: Счётчик вида sas_seed_00045 или sas_000123: формально не слово, но порядок
#: выдачи виден. Ловится разделитель, за которым идут одни цифры до конца, —
#: именно так выглядят все запрещённые примеры §5.5. Просто «кончается на три
#: цифры» проверять нельзя: у случайного ID из 36 знаков такой хвост бывает
#: примерно в двух процентах случаев, и гейт начал бы падать на честных данных.
COUNTER_PATTERN = re.compile(r"_\d{2,}$")


def check_shapes(instance_ids):
    problems = []
    for instance_id in instance_ids:
        if not ID_PATTERN.match(instance_id):
            problems.append(f"{instance_id}: форма не совпадает с sas_ + 8 знаков")
        lowered = instance_id.lower()
        for part in FORBIDDEN_PARTS:
            if part in lowered:
                problems.append(f"{instance_id}: содержит смысловой кусок «{part}»")
        if COUNTER_PATTERN.search(instance_id):
            problems.append(f"{instance_id}: похоже на порядковый счётчик")

    duplicates = [instance_id for instance_id, count
                  in Counter(instance_ids).items() if count > 1]
    problems.extend(f"{instance_id}: встречается {Counter(instance_ids)[instance_id]} раза"
                    for instance_id in duplicates)
    return problems


def _adjacent_matches(labels):
    return sum(1 for a, b in zip(labels, labels[1:]) if a == b)


def check_order(pairs, permutations=2000, seed=7):
    """Сбиваются ли метки в кучи, если отсортировать инстансы по ID.

    Возвращает ``(наблюдённые совпадения соседей, p-value, ожидание)``.
    """
    ordered = [verdict for _, verdict in sorted(pairs, key=lambda pair: pair[0])]
    observed = _adjacent_matches(ordered)

    rng = random.Random(seed)
    shuffled = list(ordered)
    at_least_as_many = 0
    total = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        matches = _adjacent_matches(shuffled)
        total += matches
        at_least_as_many += matches >= observed
    p_value = (1 + at_least_as_many) / (1 + permutations)
    return observed, p_value, total / permutations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверка §5.5: идентификаторы случайны и ничего не сообщают.")
    parser.add_argument("--inputs", required=True, nargs="+")
    parser.add_argument("--gold", default=None,
                        help="Если задан, проверяется ещё и порядок меток по ID.")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        instances = [row for path in args.inputs for row in read_jsonl(path)]
        instance_ids = [row["instance_id"] for row in instances]
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not instance_ids:
        print("ERROR: во входе нет инстансов", file=sys.stderr)
        return 2

    problems = check_shapes(instance_ids)
    warnings = []
    for problem in problems:
        print(f"id_randomness_check: {problem}", file=sys.stderr)

    order_line = ""
    if args.gold:
        try:
            gold = {row["instance_id"]: row["gold"]["verdict"]
                    for row in read_jsonl(args.gold)}
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        pairs = [(instance_id, gold[instance_id]) for instance_id in instance_ids
                 if instance_id in gold]
        if len(pairs) > 2:
            observed, p_value, expected = check_order(
                pairs, permutations=args.permutations, seed=args.seed)
            order_line = (f"порядок по ID: соседей с одной меткой {observed}, "
                          f"ожидание при случайном порядке {expected:.1f}, "
                          f"p-value {p_value:.4f}")
            if p_value < ALPHA_ORDER_FAIL:
                problems.append(
                    "метки сбиваются в кучи при сортировке по ID: идентификаторы "
                    "выданы не независимо от разметки — перегенерируйте их")
            elif p_value < ALPHA_ORDER_WARN:
                warnings.append(
                    "метки при сортировке по ID лежат теснее, чем ожидается от "
                    f"случайных идентификаторов (p={p_value:.3f}). На таком объёме "
                    "это ещё может быть совпадением, но проверьте, чем выдавались "
                    "ID: они обязаны не зависеть от разметки")

    if problems:
        print(f"id_randomness_check: НАРУШЕНИЕ — {len(problems)} проблем(ы) "
              f"на {len(instance_ids)} инстансов")
        if order_line:
            print(f"  {order_line}")
        return 1

    if warnings:
        print(f"id_randomness_check: ПОДОЗРЕНИЕ — {len(instance_ids)} инстанс(ов), "
              "форма и смысл в порядке, но порядок настораживает")
        for warning in warnings:
            print(f"  {warning}")
        if order_line:
            print(f"  {order_line}")
        return 0

    print(f"id_randomness_check: OK — {len(instance_ids)} инстанс(ов), "
          "идентификаторы случайны и ничего не сообщают")
    if order_line:
        print(f"  {order_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
