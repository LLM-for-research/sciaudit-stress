"""Общие помощники для CLI-валидаторов схем.

Сами JSON Schema лежат в корневой директории ``schemas/``. Здесь добавлены
проверки, которые одной JSON Schema не выражаются: поиск приватных полей
(утечки), дубликаты instance_id, согласованность ID evidence и сверка
предсказаний со входами.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "schemas"

FORBIDDEN_INPUT_KEYS = frozenset({
    # --- gold-метки ---
    "gold",
    "gold_verdict",
    "verdict",
    "expected_verdict",
    "supporting_eids",
    "severity",
    # --- стресс-метаданные ---
    "stress",
    "stress_type",
    "transformation_type",
    "is_stress_case",
    "seed_instance_id",
    "evidence_removal",
    "scope_expansion",
    "numeric_perturbation",
    "distractor_flag",
    # --- приватное обоснование и ревью ---
    "private_rationale",
    "rationale_private",
    "review",
    "review_note",
    "human_verification_note",
    "validation_level",
    "reviewer",
    "adjudication_note",
    "TA_note",
    # --- провенанс и лицензии ---
    "provenance",
    "provenance_map",
    "provenance_ref",
    "source_url",
    "source_ref",
    "source_location",
    "paper_title",
    "authors",
    "venue",
    "license_status",
    "is_distractor",
    # --- принадлежность к сплиту (выдаёт, в какой срез попал инстанс) ---
    "split",
    "private_slice",
    "GoldHidden",
    "AutoStressHidden",
    "ChallengeHidden",
})


def load_schema(name: str) -> dict:
    path = SCHEMAS_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[tuple[int, dict]]:
    """Вернуть [(номер_строки, объект), ...]; бросает ValueError на битом JSON."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"строка {line_no}: битый JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"строка {line_no}: ожидался объект JSON")
            rows.append((line_no, obj))
    return rows


def find_forbidden(
    obj,
    forbidden=FORBIDDEN_INPUT_KEYS,
    path="$",
    *,
    scan_values: bool = True,
) -> list[str]:
    """Рекурсивно найти запрещённые ключи *и* запрещённые строковые значения.

    Мануал требует обоих направлений: приватное имя — утечка и когда оно стоит
    ключом ``{"stress_type": ...}``, и когда значением ``{"slice":
    "AutoStressHidden"}``. По значениям сравнение точное, поэтому проза, просто
    упоминающая запрещённое слово («the authors report ...»), ворота не роняет.

    Возвращает координаты в духе JSONPath; попадание по значению печатается как
    ``$.slice == 'AutoStressHidden'``.
    """
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}"
            if key in forbidden:
                hits.append(key_path)
            elif scan_values and isinstance(value, str) and value in forbidden:
                hits.append(f"{key_path} == {value!r}")
            hits.extend(find_forbidden(value, forbidden, key_path, scan_values=scan_values))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            item_path = f"{path}[{i}]"
            if scan_values and isinstance(item, str) and item in forbidden:
                hits.append(f"{item_path} == {item!r}")
            hits.extend(find_forbidden(item, forbidden, item_path, scan_values=scan_values))
    return hits


def find_forbidden_keys(obj, forbidden=FORBIDDEN_INPUT_KEYS, path="$") -> list[str]:
    """Вариант :func:`find_forbidden` только по ключам (для тех вызовов, что
    осознанно игнорируют значения)."""
    return find_forbidden(obj, forbidden, path, scan_values=False)


def schema_errors(instance: dict, schema: dict) -> list[str]:
    """Проверить один объект по JSON Schema; вернуть строки ошибок."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        loc = "$" + "".join(
            f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path
        )
        errors.append(f"{loc}: {err.message}")
    return errors


def report(name: str, total: int, problems: list[str]) -> int:
    """Напечатать сводку; вернуть код возврата процесса (0 — ок, 1 — провал)."""
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        print(f"{name}: FAIL — ошибок {len(problems)} на {total} объект(ов)")
        return 1
    print(f"{name}: OK — {total} объект(ов), всё валидно")
    return 0
