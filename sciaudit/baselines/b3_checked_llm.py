#!/usr/bin/env python3
"""B3 — аудит с детерминированной проверкой чисел (мануал §11.5).

Тот же полный evidence pack, что у B2, и тот же вызов модели. Отличие ровно
одно: перед обращением к модели по паку прогоняется
:mod:`sciaudit.baselines.numeric_checker`, его результат кладётся в промпт, а
найденное числовое противоречие перебивает вердикт модели.

Смысл контроля тот же, что у пары B1/B2, только измеряется другой вклад. B1
против B2 показывает, что даёт ретрив; B2 против B3 — что даёт
детерминированный инструмент. Чтобы разность что-то значила, всё остальное
обязано совпадать, поэтому промпт, вызов, разбор и обработка отказов живут в
:mod:`sciaudit.baselines.model_audit`, а здесь остаётся только инструмент.

**Почему инструмент вправе перебивать модель.** Проверка чисел — это
арифметика над теми же строками пака, которые видит модель, и её результат
воспроизводим при любой температуре. Если она находит противоречие с
непустым списком eid, вердикт становится ``contradicted``: у модели нет
источника, который был бы надёжнее сложения. Обратное неверно и потому не
делается — сошедшаяся арифметика **не** превращает claim в ``warranted``,
потому что claim может быть переобобщён, вне области применимости или
опираться на несопоставимые условия. Числа умеют опровергать, но не
обосновывать.

Инструмент не применяется к безопасному отказу: если модель не ответила, B3
отказывается, как любой другой бейзлайн. Иначе сбои модели маскировались бы
детерминированным ответом и сравнение с B2 перестало бы измерять инструмент.

Запуск::

    python -m sciaudit.baselines.b3_checked_llm \\
        --input inputs.jsonl --output predictions.jsonl --model-api

Коды возврата: 0 — успех, 1 — все инстансы ушли в безопасный отказ,
2 — ошибка использования или чтения входа.
"""
from __future__ import annotations

from sciaudit.baselines import model_audit
from sciaudit.baselines.b2_fullpack_llm import select_full_pack
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
from sciaudit.baselines.numeric_checker import STATUS_FAILED, check_claim_numbers

LABEL = "B3"
#: Заглушка для system_info.model, если идентификатор не объявлен явно.
DEFAULT_MODEL_NAME = "b3-open-model"

#: Тег, которым помечается перебитый вердикт. Он же в словаре схемы предсказания.
NUMERIC_TAG = "numerical_inconsistency"

#: Уверенность перебитого вердикта. Арифметика детерминирована, поэтому число
#: высокое — но не 1.0: чекер разбирает числа регулярными выражениями и может
#: не так понять таблицу. Значение осознанно единое: если инструмент ошибается,
#: это обязано быть видно в selective risk, а не размазываться по шкале.
TOOL_CONFIDENCE = 0.9


def _render(results) -> str:
    lines = []
    for r in results:
        eids = ", ".join(r["eids"]) if r["eids"] else "no evidence ids"
        lines.append(f'- "{r["claim_fragment"]}" -> {r["status"].upper()}: '
                     f'{r["reason"]} [{eids}]')
    return "\n".join(lines)


class NumericTool:
    """Детерминированная проверка чисел как инструмент бейзлайна.

    Держит два решения B3 и больше ничего: что сказать модели и когда её
    поправить. Состояния между инстансами нет — один и тот же объект можно
    переиспользовать на всём прогоне.
    """

    def __init__(self, override=True, confidence=TOOL_CONFIDENCE):
        self.override = override
        self.confidence = confidence

    def check(self, claim_text, units):
        return check_claim_numbers(claim_text, list(units))

    def brief(self, claim_text, units):
        """Секция промпта с результатом проверки. ``None``, если проверять нечего."""
        results = self.check(claim_text, units)
        if not results:
            return None
        return (
            "Deterministic numeric check of this claim against the evidence above.\n"
            "It was computed by exact arithmetic, not by a language model. Where it\n"
            "reports FAILED, the numbers contradict the claim; trust it over your own\n"
            "arithmetic. Where it reports UNKNOWN, the pack has nothing to compare and\n"
            "the check says nothing either way.\n"
            f"{_render(results)}\n"
        )

    def refine(self, prediction, claim_text, units):
        """Перебить вердикт, если проверка нашла числовое противоречие.

        Правится ровно четыре поля: вердикт, теги, evidence и уверенность. Всё
        остальное, включая обоснование модели, остаётся её ответом — иначе по
        файлу предсказаний нельзя будет понять, что именно сделала модель, а
        что инструмент.
        """
        prediction["system_info"]["uses_numeric_checker"] = True
        if not self.override:
            return prediction

        results = self.check(claim_text, units)
        contradictions = [r for r in results
                          if r["status"] == STATUS_FAILED and r["eids"]]
        if not contradictions:
            return prediction

        allowed = {u.get("eid") for u in units if u.get("eid")}
        tool_eids = [e for r in contradictions for e in r["eids"] if e in allowed]

        prediction["verdict"] = "contradicted"
        prediction["abstain"] = False
        prediction["confidence"] = max(prediction.get("confidence", 0.0), self.confidence)

        tags = list(prediction.get("issue_tags") or [])
        if NUMERIC_TAG not in tags:
            tags.append(NUMERIC_TAG)
        prediction["issue_tags"] = tags

        eids = list(prediction.get("predicted_eids") or [])
        for eid in tool_eids:
            if eid not in eids:
                eids.append(eid)
        prediction["predicted_eids"] = eids

        reason = contradictions[0]["reason"]
        prediction["rationale_short"] = (
            f"Numeric check overrode the model verdict: {reason}"
        )[:500]
        return prediction


def run(input_path, output_path, **kwargs):
    """Прогнать B3. Сигнатура совпадает с B2; инструмент подставляется здесь."""
    kwargs.setdefault("tool", NumericTool())
    return model_audit.run(input_path, output_path, select_full_pack, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        "Бейзлайн B3: полный evidence pack плюс детерминированная проверка чисел.",
        default_model_name=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--no-override", action="store_true",
        help="Показывать модели результат проверки, но не перебивать её вердикт. "
             "Разделяет два вклада инструмента: подсказку и жёсткое правило.")
    args = parser.parse_args(argv)
    return run_cli(args, select_full_pack, LABEL,
                   tool=NumericTool(override=not args.no_override))


if __name__ == "__main__":
    raise SystemExit(main())
