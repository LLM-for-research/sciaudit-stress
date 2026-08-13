#!/usr/bin/env python3
"""Рекурсивный скан запрещённых ключей по файлам, видимым участникам (мануал §10.1).

``validate_inputs`` проверяет один входной файл Track A по одной схеме. Это —
более широкие ворота: они обходят дерево директорий, разбирают каждый понятный
им структурированный файл и падают, если приватный материал (gold-метки,
стресс-метаданные, провенанс, принадлежность к сплиту) оказался там, где его
может прочитать участник. Пункт обязателен в чеклисте запуска (§19.5).

**Роли, а не одно правило на все файлы.** Публичный ``gold.jsonl`` законно несёт
``verdict`` и ``supporting_eids`` (Listing 5); файл предсказаний законно несёт
собственный ``verdict`` системы (Listing 3). Скан обоих по правилу для входов
падал бы на корректных данных, а ворота, которые кричат «волки», рано или поздно
выключают. Поэтому каждому файлу по его пути присваивается *профиль*, и профиль
решает, какое подмножество ``FORBIDDEN_INPUT_KEYS`` к нему применяется:

===========  ====================================================================
input        запрещено всё (по умолчанию и как fail-closed запасной вариант)
gold         сама метка ожидаема; стресс, провенанс и сплит — нет
prediction   ожидаем собственный ``verdict`` системы, больше ничего
internal     участникам не виден вовсе — пропускается с печатью причины
===========  ====================================================================

Каждое исключение — строка в ``DEFAULT_RULES`` ниже с письменной причиной, так
что ослабление ворот требует отревьюенного диффа, а не тихого флага.

Запуск::

    python -m sciaudit.leakage.forbidden_key_scan data_public/ examples/
    python -m sciaudit.leakage.forbidden_key_scan data_public/ --profile input

Коды возврата: 0 — чисто, 1 — найдена утечка, 2 — ошибка использования или разбора.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

from sciaudit.schemas import FORBIDDEN_INPUT_KEYS, find_forbidden

# Поля, которые *публичный gold* несёт по построению (мануал, Listing 5).
GOLD_ALLOWED = frozenset({"gold", "gold_verdict", "verdict", "supporting_eids"})
# Предсказание — выход системы; его вердикт и есть то, что измеряется.
PREDICTION_ALLOWED = frozenset({"verdict"})

#: имя профиля -> запрещённый набор (``None`` — «участникам не виден, пропустить»)
PROFILES: dict[str, frozenset[str] | None] = {
    "input": FORBIDDEN_INPUT_KEYS,
    "gold": FORBIDDEN_INPUT_KEYS - GOLD_ALLOWED,
    "prediction": FORBIDDEN_INPUT_KEYS - PREDICTION_ALLOWED,
    "internal": None,
}

DEFAULT_PROFILE = "input"

#: (glob, профиль, причина). Побеждает первое совпадение. Glob со слешем
#: сопоставляется с путём относительно корня скана, иначе — с именем файла.
DEFAULT_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "*internal_annotation*",
        "internal",
        "пример приватной схемы; показывает запись аннотации, участникам не отдаётся никогда",
    ),
    (
        "toy_stress_cases.jsonl",
        "internal",
        "учебный материал по методологии стресса (docs/stress_templates.md); "
        "в data_public/ попадает только публичная половина каждого кейса",
    ),
    ("*gold*.json", "gold", "публичный gold-файл (Listing 5)"),
    ("*gold*.jsonl", "gold", "публичный gold-файл (Listing 5)"),
    ("*predictions*.jsonl", "prediction", "выход системы, а не вход для участника (Listing 3)"),
    ("*preds*.jsonl", "prediction", "выход системы, а не вход для участника (Listing 3)"),
)

STRUCTURED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".yaml", ".yml"}


class ScanError(Exception):
    """Файл не удалось разобрать — это отказ, а не пропуск."""


# --- файл -> профиль ---------------------------------------------------------

def _matches(pattern: str, rel_posix: str, name: str) -> bool:
    target = rel_posix if "/" in pattern else name
    return fnmatch.fnmatch(target, pattern)


def classify(path: Path, root: Path, rules=DEFAULT_RULES) -> tuple[str, str]:
    """Вернуть ``(профиль, причина)`` для одного файла."""
    try:
        rel_posix = path.relative_to(root).as_posix()
    except ValueError:
        rel_posix = path.as_posix()
    for pattern, profile, reason in rules:
        if _matches(pattern, rel_posix, path.name):
            return profile, reason
    return DEFAULT_PROFILE, "по умолчанию: считается видимым участнику входом"


def load_rules(path: Path) -> tuple[tuple[str, str, str], ...]:
    """Загрузить файл переопределения правил: ``[{"glob":…, "profile":…, "reason":…}, …]``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ScanError(f"{path}: не читается файл правил: {e}") from e
    rules = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "glob" not in item or "profile" not in item:
            raise ScanError(f"{path}: правилу {i} нужны как минимум 'glob' и 'profile'")
        if item["profile"] not in PROFILES:
            raise ScanError(f"{path}: у правила {i} неизвестный профиль {item['profile']!r}")
        rules.append((item["glob"], item["profile"], item.get("reason", "(причина не указана)")))
    return tuple(rules)


# --- разбор -----------------------------------------------------------------

def parse_objects(path: Path) -> list[tuple[str, object]]:
    """Вернуть ``[(метка_места, разобранный_объект), …]`` для структурированного файла."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")

    if suffix in {".jsonl", ".ndjson"}:
        objects = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                objects.append((f":{line_no}", json.loads(line)))
            except json.JSONDecodeError as e:
                raise ScanError(f"{path}:{line_no}: битый JSON: {e}") from e
        return objects

    if suffix == ".json":
        try:
            return [("", json.loads(text))]
        except json.JSONDecodeError as e:
            raise ScanError(f"{path}: битый JSON: {e}") from e

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            return []  # формат необязательный; отсутствие PyYAML не должно имитировать успех
        try:
            return [("", doc) for doc in yaml.safe_load_all(text) if doc is not None]
        except yaml.YAMLError as e:
            raise ScanError(f"{path}: битый YAML: {e}") from e

    return []


def iter_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.is_file())


# --- сканирование ------------------------------------------------------------

def scan_file(
    path: Path,
    root: Path,
    rules=DEFAULT_RULES,
    forced_profile: str | None = None,
) -> tuple[str, str, list[str]]:
    """Просканировать один файл. Возвращает ``(профиль, причина, утечки)``."""
    if forced_profile:
        profile, reason = forced_profile, "задан принудительно через --profile"
    else:
        profile, reason = classify(path, root, rules)

    forbidden = PROFILES[profile]
    if forbidden is None:
        return profile, reason, []

    leaks = []
    for label, obj in parse_objects(path):
        for hit in find_forbidden(obj, forbidden):
            leaks.append(f"{path}{label}: [{profile}] {hit}")
    return profile, reason, leaks


def scan(
    targets: list[Path],
    rules=DEFAULT_RULES,
    forced_profile: str | None = None,
    verbose: bool = False,
) -> tuple[list[str], int]:
    """Просканировать файлы и директории. Возвращает ``(утечки, число_файлов)``."""
    leaks: list[str] = []
    examined = 0

    for target in targets:
        if not target.exists():
            raise ScanError(f"{target}: нет такого файла или директории")
        root = target if target.is_dir() else target.parent

        for path in iter_files(target):
            if path.suffix.lower() not in STRUCTURED_SUFFIXES:
                continue
            examined += 1
            profile, reason, file_leaks = scan_file(path, root, rules, forced_profile)
            leaks.extend(file_leaks)
            if verbose:
                if PROFILES[profile] is None:
                    print(f"  SKIP  {path} [{profile}] — {reason}")
                else:
                    status = "LEAK" if file_leaks else "ok"
                    print(f"  {status:5} {path} [{profile}]")

    return leaks, examined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Поиск просочившихся приватных полей в файлах, видимых участникам (§10.1).",
    )
    parser.add_argument("paths", nargs="+", help="Файлы или директории для сканирования.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="Задать один профиль для всех файлов вместо вывода профиля из пути.",
    )
    parser.add_argument("--rules", type=Path, help="JSON-файл, переопределяющий встроенные правила.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Перечислить все просмотренные файлы.")
    args = parser.parse_args(argv)

    try:
        rules = load_rules(args.rules) if args.rules else DEFAULT_RULES
        leaks, examined = scan(
            [Path(p) for p in args.paths], rules, args.profile, args.verbose
        )
    except ScanError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if leaks:
        for leak in leaks:
            print(f"LEAK: {leak}")
        print(
            f"forbidden_key_scan: FAIL — просочившихся полей {len(leaks)} "
            f"в {examined} файл(ах)"
        )
        return 1

    print(f"forbidden_key_scan: OK — {examined} файл(ов), чисто")
    return 0


if __name__ == "__main__":
    sys.exit(main())
