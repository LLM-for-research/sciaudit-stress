#!/usr/bin/env python3
"""B2 — аудит по полному evidence pack, научный контроль к B1 (мануал §11.4).

Тот же вызов модели, что и в B1, но без ретрива: в промпт подаётся весь
evidence pack инстанса. Смысл — измерить вклад BM25. Если B2 не хуже B1,
ретрив в текущем виде не помогает; если B2 заметно лучше, BM25 выбрасывает
решающие единицы; если B2 хуже, длинный пак топит модель шумом.

Чтобы сравнение что-то значило, всё, кроме выбора evidence, у B1 и B2 обязано
совпадать. Поэтому промпт, вызов, разбор, нормализация и безопасный отказ живут
в :mod:`sciaudit.baselines.model_audit`, а здесь остаётся ровно селектор.

Запуск::

    python -m sciaudit.baselines.b2_fullpack_llm \\
        --input inputs.jsonl --output predictions.jsonl \\
        --model-command "your-open-model-command" \\
        --model-name "your-open-model-id"

Коды возврата: 0 — успех, 1 — все инстансы ушли в безопасный отказ,
2 — ошибка использования или чтения входа.
"""
from __future__ import annotations

from sciaudit.baselines.model_audit import (  # noqa: F401  (публичная поверхность модуля)
    DEFAULT_TIMEOUT_SECONDS,
    FALLBACK_RATIONALE,
    ISSUE_TAGS,
    MOCK_MODEL_NAME,
    ModelCallError,
    build_arg_parser,
    count_fallbacks,
    run_cli,
)
from sciaudit.baselines import model_audit

LABEL = "B2"
#: Заглушка для system_info.model, если идентификатор не объявлен явно.
DEFAULT_MODEL_NAME = "b2-open-model"


def select_full_pack(claim_text, evidence_pack):
    """Селектор B2: весь пак, в исходном порядке, без ранжирования."""
    return list(evidence_pack)


def run(input_path, output_path, **kwargs):
    """Прогнать B2. Сигнатура совпадает с B1 за вычетом ``top_k``."""
    return model_audit.run(input_path, output_path, select_full_pack, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        "Бейзлайн B2: аудит по полному evidence pack, без ретрива.",
        default_model_name=DEFAULT_MODEL_NAME)
    args = parser.parse_args(argv)
    return run_cli(args, select_full_pack, LABEL)


if __name__ == "__main__":
    raise SystemExit(main())
