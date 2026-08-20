#!/usr/bin/env python3
"""B4 — бейзлайн осторожного отказа (мануал §11.6).

Единственный бейзлайн, которому разрешено не отвечать. B0 задаёт пол, B1 против
B2 меряет вклад ретрива, B2 против B3 — вклад детерминированного инструмента, а
B4 меряет **цену осторожности**: сколько покрытия приходится отдать, чтобы упали
тяжёлые промахи. Без него половина метрик evaluator'а — покрытие, кривая
риск–покрытие, ``coverage_at_target_sfwr``, отказ в разрезе gold-вердикта —
проверяется только на тестах, потому что все остальные бейзлайны отвечают всегда.

Пять правил отказа взяты из §11.6 дословно:

1. ``unparsable_json`` — ответ не разобрался после повторов;
2. ``low_confidence`` — заявленная моделью уверенность ниже порога;
3. ``empty_evidence`` — решительный вердикт (``warranted`` / ``overclaimed`` /
   ``contradicted``) назван без единого eid;
4. ``checker_conflict`` — детерминированная проверка чисел противоречит
   вердикту модели;
5. ``ensemble_split`` — голоса ансамбля разошлись.

Правило 4 — то самое место, где B4 расходится с B3, и расхождение
содержательное. B3 считает, что арифметика надёжнее модели, и **перебивает**
вердикт. B4 считает, что расхождение инструмента с моделью — признак того, что
инстанс тяжёлый, и **отказывается**. Обе позиции защитимы; ради того, чтобы их
можно было сравнить числом, они разведены по разным бейзлайнам.

Правило 5 опирается на прямое измерение: два одинаковых прогона B2 на
``public_dev`` разошлись в 6 вердиктах из 24 (``docs/baselines_compared.md``). Там же, где
голоса совпали, accuracy оказалась 0.778, где разошлись — 0.333. То есть
собственная нестабильность модели предсказывает её ошибку, и B4 обращает
недетерминизм эндпоинта из помехи в сигнал отказа.

Уверенность B4 — произведение двух независимых сигналов: доли согласия голосов
и средней заявленной уверенности победившего вердикта. При ``--votes 1``
согласие тождественно единице, и остаётся ровно уверенность модели.

Запуск::

    python -m sciaudit.baselines.b4_selective \\
        --input inputs.jsonl --output predictions.jsonl --model-api --votes 3

Коды возврата: 0 — успех, 1 — все инстансы ушли в безопасный отказ,
2 — ошибка использования или чтения входа.
"""
from __future__ import annotations

import sys
from collections import Counter

from sciaudit.baselines import model_audit
from sciaudit.baselines.b2_fullpack_llm import select_full_pack
from sciaudit.baselines.b3_checked_llm import NumericTool
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

LABEL = "B4"
#: Заглушка для system_info.model, если идентификатор не объявлен явно.
DEFAULT_MODEL_NAME = "b4-open-model"

#: Сколько раз спросить модель об одном инстансе. Три — минимум, при котором
#: существует большинство и его отсутствие: два голоса дают только «согласны» и
#: «не согласны», без градаций.
DEFAULT_VOTES = 3

#: Порог заявленной уверенности. Выбран по распределению из живого прогона B2 на
#: public_dev: модель почти не пользуется нижней половиной шкалы (все 24 ответа
#: легли в 0.73–0.99), поэтому пороги вроде 0.5 не срабатывают никогда.
#: ВАЖНО: порог подобран на том же срезе, на котором B4 потом измеряется, так что
#: его числа на public_dev оптимистичны. Честная калибровка требует отдельного
#: среза — см. «скрытый сплит» в плане работ.
DEFAULT_CONFIDENCE_THRESHOLD = 0.9

#: Вердикты, которые обязаны опираться на evidence. ``insufficient`` — не обязан:
#: утверждение «доказательств не хватает» как раз и означает, что назвать нечего.
DECISIVE_VERDICTS = frozenset({"warranted", "overclaimed", "contradicted"})

RULE_UNPARSABLE = "unparsable_json"
RULE_LOW_CONFIDENCE = "low_confidence"
RULE_EMPTY_EVIDENCE = "empty_evidence"
RULE_CHECKER_CONFLICT = "checker_conflict"
RULE_ENSEMBLE_SPLIT = "ensemble_split"

#: Порядок отчёта, он же порядок §11.6. Отказ печатает все сработавшие правила, а
#: не первое: инстанс, где сошлись три причины, отличается от инстанса с одной.
RULES = (RULE_UNPARSABLE, RULE_LOW_CONFIDENCE, RULE_EMPTY_EVIDENCE,
         RULE_CHECKER_CONFLICT, RULE_ENSEMBLE_SPLIT)


def _majority(verdicts):
    """Победивший вердикт и доля голосов за него.

    Ничья разрешается детерминированно — по алфавиту, — потому что при ничьей
    правило ``ensemble_split`` всё равно сработает, и выбор влияет только на то,
    какой вердикт будет назван в обосновании отказа.
    """
    counts = Counter(verdicts)
    best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return best[0], best[1]


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class SelectiveAudit:
    """Ансамбль голосов и правила отказа поверх них.

    Подставляется в общий цикл :func:`sciaudit.baselines.model_audit.run` через
    шов ``audit``, поэтому чтение входа, запись предсказаний и счётчик
    безопасных отказов у B4 те же, что у остальных бейзлайнов.

    Объект накапливает статистику по правилам (``counts``) — без неё по файлу
    предсказаний нельзя понять, чем именно вызван отказ, а значит нельзя и
    сказать, какое из пяти правил работает.
    """

    def __init__(self, votes=DEFAULT_VOTES,
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                 require_unanimous=False, disabled_rules=()):
        if votes < 1:
            raise ValueError("--votes должно быть не меньше 1.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("--confidence-threshold должен лежать в [0, 1].")
        self.votes = votes
        self.confidence_threshold = confidence_threshold
        self.require_unanimous = require_unanimous
        self.disabled = frozenset(disabled_rules)
        self.counts = Counter()

    # --- правила -------------------------------------------------------------

    def _fired(self, verdict, agreement, mean_confidence, prediction, numeric):
        fired = []
        if agreement <= 0.5 or (self.require_unanimous and agreement < 1.0):
            fired.append(RULE_ENSEMBLE_SPLIT)
        if mean_confidence < self.confidence_threshold:
            fired.append(RULE_LOW_CONFIDENCE)
        if verdict in DECISIVE_VERDICTS and not prediction.get("predicted_eids"):
            fired.append(RULE_EMPTY_EVIDENCE)
        if verdict != "contradicted" and any(
                r["status"] == STATUS_FAILED and r["eids"] for r in numeric):
            fired.append(RULE_CHECKER_CONFLICT)
        return [rule for rule in RULES if rule in fired and rule not in self.disabled]

    # --- один инстанс --------------------------------------------------------

    def __call__(self, instance, select_evidence, **kwargs):
        ballots = [model_audit.audit_instance(instance, select_evidence, **kwargs)
                   for _ in range(self.votes)]
        self.counts["votes"] += len(ballots)

        parsed = [pred for pred, reason in ballots if reason is None]
        if not parsed:
            # Модель не ответила ни разу. Возвращается нетронутый безопасный
            # отказ: его обоснование — то самое, по которому считается доля
            # сбоев, и переписывать его значило бы прятать сбой за осторожностью.
            self.counts["abstained"] += 1
            self.counts[RULE_UNPARSABLE] += 1
            return ballots[-1]

        verdict, wins = _majority(pred["verdict"] for pred in parsed)
        agreement = wins / len(ballots)
        winning = [pred for pred in parsed if pred["verdict"] == verdict]
        mean_confidence = _mean(pred["confidence"] for pred in winning)

        claim_text = model_audit.claim_text_of(instance)
        units = select_evidence(claim_text, instance.get("evidence_pack", []))
        numeric = check_claim_numbers(claim_text, list(units))

        prediction = dict(winning[0])
        prediction["confidence"] = round(agreement * mean_confidence, 4)

        fired = self._fired(verdict, agreement, mean_confidence, prediction, numeric)
        if not fired:
            self.counts["answered"] += 1
            prediction["abstain"] = False
            return prediction, None

        self.counts["abstained"] += 1
        for rule in fired:
            self.counts[rule] += 1

        # Отказ выражается по общей конвенции проекта: консервативный вердикт
        # плюс abstain. Вердикт большинства не теряется — он назван в
        # обосновании, иначе по файлу нельзя восстановить, от чего отказались.
        prediction["verdict"] = model_audit.SAFE_FALLBACK_VERDICT
        prediction["abstain"] = True
        prediction["rationale_short"] = (
            f"B4 abstained ({', '.join(fired)}); majority vote was {verdict} "
            f"at agreement {agreement:.2f}, stated confidence {mean_confidence:.2f}."
        )[:500]
        return prediction, None

    # --- отчёт ---------------------------------------------------------------

    def summary(self):
        answered = self.counts["answered"]
        abstained = self.counts["abstained"]
        total = answered + abstained
        parts = [f"вызовов модели — {self.counts['votes']}",
                 f"отвечено — {answered}/{total}",
                 f"отказов — {abstained}"]
        by_rule = ", ".join(f"{rule} {self.counts[rule]}"
                            for rule in RULES if self.counts[rule])
        if by_rule:
            parts.append(f"правила: {by_rule}")
        return "; ".join(parts)


def run(input_path, output_path, votes=DEFAULT_VOTES,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        require_unanimous=False, **kwargs):
    """Прогнать B4. Пак тот же, что у B2 и B3; отличается способ решать."""
    kwargs.setdefault("tool", NumericTool(override=False))
    kwargs.setdefault("audit", SelectiveAudit(
        votes=votes, confidence_threshold=confidence_threshold,
        require_unanimous=require_unanimous))
    return model_audit.run(input_path, output_path, select_full_pack, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        "Бейзлайн B4: полный evidence pack, ансамбль голосов и пять правил отказа.",
        default_model_name=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--votes", type=int, default=DEFAULT_VOTES,
        help=f"Сколько раз спросить модель об одном инстансе (по умолчанию "
             f"{DEFAULT_VOTES}). При 1 правило ensemble_split не работает, и "
             f"стоимость прогона совпадает с B2.")
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Ниже этой заявленной уверенности бейзлайн отказывается "
             f"(по умолчанию {DEFAULT_CONFIDENCE_THRESHOLD}).")
    parser.add_argument(
        "--require-unanimous", action="store_true",
        help="Отказываться при любом расхождении голосов, а не только при "
             "отсутствии большинства. Более осторожная точка на кривой.")
    args = parser.parse_args(argv)

    try:
        audit = SelectiveAudit(votes=args.votes,
                               confidence_threshold=args.confidence_threshold,
                               require_unanimous=args.require_unanimous)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    status = run_cli(args, select_full_pack, LABEL,
                     tool=NumericTool(override=False), audit=audit)
    if audit.counts["votes"]:
        print(f"{LABEL}: {audit.summary()}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
