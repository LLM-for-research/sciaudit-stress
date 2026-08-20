#!/usr/bin/env python3
"""Перевыдача идентификаторов инстансов (мануал §5.5).

Мануал требует случайных несемантических ID и называет перегенерацию штатным
лекарством, когда проба §10.2 или проверка §5.5 находит связь идентификатора с
разметкой. Этот модуль — то самое лекарство.

Что делается за один проход:

* каждому инстансу выдаётся новый ID вида ``sas_`` плюс восемь знаков из
  криптографического источника;
* ссылки ``stress.seed_instance_id`` переписываются на новые ID, иначе
  стресс-вариант потеряет свой seed и §10.4 станет непроверяемым;
* порядок записей перемешивается — чтобы порядок строк в файле тоже перестал
  что-либо сообщать о том, в каком порядке шла разметка;
* та же замена применяется к файлам из ``--rewrite`` (манифесты, заметки), где
  старые ID встречаются текстом;
* полученная выдача проверяется тем же тестом, что и гейт §5.5, и при
  подозрительной раскладке жребий бросается заново.

Последний пункт — не подгонка под тест. Случайная выдача изредка складывается
так, что метки при сортировке по ID сбиваются в кучи: у двадцати четырёх
инстансов это примерно раз на полтора сотни попыток. Такая выдача ничего не
сообщает о разметке — но выглядит ровно как та, что сообщает, и будет вечно
ронять гейт. Отбрасываются не «плохие» ID, а случайное совпадение; связи с
метками ни один из вариантов не несёт, потому что источник случайности их не
видит.

Публичные ``inputs.jsonl`` и ``gold.jsonl`` здесь **не трогаются**: они
выводятся из приватных записей проекцией, поэтому после перегенерации нужно
заново прогнать :mod:`sciaudit.construction.derive_public`. Иначе появилось бы
второе место, где публичный срез собирается, — и однажды они разошлись бы.

Запуск::

    python -m sciaudit.construction.regenerate_ids \\
        --internal data_paper_derived/*.internal_annotation.jsonl \\
        --rewrite data_public/public_dev/manifest.json \\
        --map-out /tmp/id_map.json

По умолчанию источник случайности — :mod:`secrets`, и результат каждый раз
разный. ``--seed`` делает прогон воспроизводимым и нужен только тестам: в
настоящей перегенерации воспроизводимость — это ровно то свойство, от которого
мы избавляемся.

Коды возврата: 0 — перевыдано, 2 — ошибка использования.
"""
from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
PREFIX = "sas_"
LENGTH = 8


def new_id(taken, rng=None):
    """Свежий ID, которого ещё нет ни у кого."""
    while True:
        pick = rng.choice if rng else secrets.choice
        candidate = PREFIX + "".join(pick(ALPHABET) for _ in range(LENGTH))
        if candidate not in taken:
            taken.add(candidate)
            return candidate


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")


def build_mapping(records, reserved=(), rng=None):
    """Старый ID → новый, для всех записей сразу.

    Одним словарём на весь корпус, а не по файлам: ``seed_instance_id`` может
    указывать на инстанс из другого файла, и переписывать ссылки можно только
    зная все замены.
    """
    taken = set(reserved)
    mapping = {}
    for record in records:
        instance_id = record["instance_id"]
        if instance_id in mapping:
            raise ValueError(f"дубликат instance_id {instance_id}")
        mapping[instance_id] = new_id(taken, rng)
    return mapping


def apply_mapping(record, mapping):
    record = dict(record)
    record["instance_id"] = mapping[record["instance_id"]]

    stress = record.get("stress")
    if isinstance(stress, dict) and stress.get("seed_instance_id"):
        seed = stress["seed_instance_id"]
        if seed not in mapping:
            raise ValueError(
                f"{record['instance_id']}: seed {seed} не найден среди "
                "перевыдаваемых инстансов — ссылка повисла бы")
        stress = dict(stress)
        stress["seed_instance_id"] = mapping[seed]
        record["stress"] = stress
    return record


def rewrite_text(path, mapping):
    """Заменить старые ID в произвольном текстовом файле.

    Замена литеральная: ID — уникальный двенадцатизначный токен, случайно
    совпасть с другим текстом он не может.
    """
    text = Path(path).read_text(encoding="utf-8")
    hits = 0
    for old, new in mapping.items():
        if old in text:
            hits += text.count(old)
            text = text.replace(old, new)
    Path(path).write_text(text, encoding="utf-8")
    return hits


#: Сколько раз перебросить жребий, если раскладка вышла подозрительной.
DEFAULT_MAX_ATTEMPTS = 20


def order_is_unremarkable(records, mapping, gold, permutations=2000):
    """Не сбиваются ли метки в кучи при сортировке по новым ID.

    Возвращает ``(в порядке, p-value)``. Без gold проверять нечего — тогда
    считаем, что всё в порядке: перевыдача не обязана иметь доступ к меткам.
    """
    if not gold:
        return True, None
    pairs = [(mapping[record["instance_id"]], gold[record["instance_id"]])
             for record in records if record["instance_id"] in gold]
    if len(pairs) < 4:
        return True, None
    from sciaudit.leakage.id_randomness_check import ALPHA_ORDER_WARN, check_order
    _, p_value, _ = check_order(pairs, permutations=permutations)
    return p_value >= ALPHA_ORDER_WARN, p_value


def read_gold(path):
    return {row["instance_id"]: row["gold"]["verdict"] for row in read_jsonl(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Перевыдать инстансам случайные идентификаторы (§5.5).")
    parser.add_argument("--internal", nargs="+", required=True)
    parser.add_argument("--rewrite", nargs="*", default=(),
                        help="Файлы, где старые ID встречаются текстом: манифесты, "
                             "заметки. Публичные inputs/gold сюда не входят — их "
                             "надо заново вывести derive_public.")
    parser.add_argument("--avoid", nargs="*", default=(),
                        help="JSONL других сплитов: их ID не будут выданы повторно.")
    parser.add_argument("--map-out", default=None,
                        help="Куда записать соответствие старых ID новым.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Только для тестов: делает перевыдачу воспроизводимой.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Не перемешивать порядок записей в файлах.")
    parser.add_argument("--gold", default=None,
                        help="Публичный gold текущего среза. Если задан, выдача "
                             "проверяется тем же тестом, что и гейт §5.5, и "
                             "подозрительная раскладка перебрасывается.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed) if args.seed is not None else None

    try:
        per_file = {path: read_jsonl(path) for path in args.internal}
        records = [record for rows in per_file.values() for record in rows]
        if not records:
            raise ValueError("во входе нет записей")

        reserved = set()
        for path in args.avoid:
            reserved |= {row["instance_id"] for row in read_jsonl(path)}

        gold = read_gold(args.gold) if args.gold else {}

        attempts = 0
        while True:
            attempts += 1
            mapping = build_mapping(records, reserved, rng)
            ok, p_value = order_is_unremarkable(records, mapping, gold)
            if ok or attempts >= args.max_attempts:
                break
            print(f"regenerate_ids: попытка {attempts} отброшена — метки при "
                  f"сортировке по ID легли в кучи (p={p_value:.3f}), "
                  "бросаю жребий заново", file=sys.stderr)
        if gold and p_value is not None:
            print(f"regenerate_ids: раскладка меток по ID неотличима от случайной "
                  f"(p={p_value:.3f}), попыток: {attempts}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        updated = {path: [apply_mapping(record, mapping) for record in rows]
                   for path, rows in per_file.items()}
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.no_shuffle:
        shuffler = rng or random.SystemRandom()
        for rows in updated.values():
            shuffler.shuffle(rows)

    if args.dry_run:
        print(f"regenerate_ids: dry-run — перевыдано было бы {len(mapping)} ID")
        return 0

    for path, rows in updated.items():
        write_jsonl(path, rows)
        print(f"regenerate_ids: {path} — {len(rows)} запис(ей) перевыдано")

    for path in args.rewrite:
        hits = rewrite_text(path, mapping)
        print(f"regenerate_ids: {path} — заменено вхождений: {hits}")

    if args.map_out:
        Path(args.map_out).write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"regenerate_ids: соответствие записано в {args.map_out}")

    print("regenerate_ids: теперь заново выведите публичный срез — "
          "python -m sciaudit.construction.derive_public")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
