"""Evaluator Track A: сравнение предсказаний с приватным gold (мануал §12).

Считает качество вердикта, локализацию evidence, issue-теги, severe false-warrant
rate (§3.4) и selective risk (§3.5, §12.5). Пишет метрики в JSON и читаемый
отчёт в Markdown.

Два свойства, без которых числа не значат ничего.

**Сабмишен обязан быть полным.** Ровно одно предсказание на каждый gold-инстанс:
без пропусков, дубликатов и лишних ID. Иначе систему можно «улучшить», прислав
только те ответы, в которых она уверена: пропущенные инстансы не попадали в
знаменатель, и accuracy росла от молчания. Неуверенность выражается явным
``abstain: true`` — тогда инстанс остаётся в знаменателе покрытия, а качество
считается отдельно по отвеченным. Неполный сабмишен не скорится: метрики
качества становятся ``null``, а CLI возвращает ненулевой код.

**Кривая risk-coverage строится ранжированием, а не фиксированными порогами.**
При фиксированной сетке порогов система, поставившая всем ответам
``confidence = 1.0``, даёт одинаковое покрытие во всех точках, площадь под
кривой вырождается, и полностью неверная система получает AURC = 0. Поэтому
ответы сортируются по убыванию уверенности, покрытие пробегает k/N, а внутри
групп с равной уверенностью счётчики интерполируются — результат не зависит от
порядка внутри группы. Дополнительно считается AUGRC: обобщённый риск нормируется
на все N инстансов, а не на выбранные, поэтому область малого покрытия не
получает непропорционального веса.

Запуск:
    python -m sciaudit.evaluator.score --pred preds.jsonl --gold gold.jsonl \\
        --out metrics.json --report report.md
"""
import argparse
import json
import math
from pathlib import Path

from sciaudit.schemas import read_jsonl as _read_jsonl_with_line_numbers

VERDICTS = ["warranted", "overclaimed", "contradicted", "insufficient"]
TARGET_SFWR_LEVELS = [0.0, 0.05, 0.1, 0.2]
FIXED_COVERAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def read_jsonl(path):
    """Объекты JSONL без номеров строк.

    Парсер один на весь репозиторий — ``sciaudit.schemas.read_jsonl``: он же
    проверяет, что каждая строка является объектом. Здесь только снимается
    нумерация и в текст ошибки добавляется путь к файлу.
    """
    try:
        return [obj for _, obj in _read_jsonl_with_line_numbers(path)]
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def validate_prediction(pred):
    # Список обязан совпадать с `required` в schemas/prediction.schema.json.
    # Там `abstain` необязателен — система, которая никогда не отказывается от
    # ответа, вправе его не писать (мануал, Listing 3). Если требовать его
    # здесь, схема-валидные прогнозы молча выпадают из знаменателя вместо того,
    # чтобы быть оценёнными.
    required = [
        "instance_id",
        "verdict",
        "confidence",
        "predicted_eids",
        "issue_tags",
    ]

    missing = [k for k in required if k not in pred]
    if missing:
        return [f"нет обязательных полей: {missing}"]

    errors = []

    if pred["verdict"] not in VERDICTS:
        errors.append(f"недопустимый verdict: {pred['verdict']}")

    if not isinstance(pred["confidence"], (int, float)) or not 0 <= pred["confidence"] <= 1:
        errors.append("confidence должен быть числом в [0, 1]")

    if not isinstance(pred["predicted_eids"], list):
        errors.append("predicted_eids должен быть списком")

    if not isinstance(pred["issue_tags"], list):
        errors.append("issue_tags должен быть списком")

    if "abstain" in pred and not isinstance(pred["abstain"], bool):
        errors.append("abstain должен быть булевым")

    return errors


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def normalize_gold(row):
    gold = row.get("gold", row)
    return {
        "verdict": gold["verdict"],
        "supporting_eids": gold.get("supporting_eids", []),
        "issue_tags": gold.get("issue_tags", []),
    }


def get_non_abstained_ids(golds, preds):
    return [
        instance_id
        for instance_id in golds
        if instance_id in preds and not preds[instance_id].get("abstain", False)
    ]


def verdict_metrics(golds, preds, instance_ids):
    correct = 0
    total = len(instance_ids)
    per_class = {}

    for label in VERDICTS:
        tp = fp = fn = 0

        for instance_id in instance_ids:
            gold_label = golds[instance_id]["verdict"]
            pred_label = preds[instance_id]["verdict"]

            if pred_label == label and gold_label == label:
                tp += 1
            elif pred_label == label and gold_label != label:
                fp += 1
            elif pred_label != label and gold_label == label:
                fn += 1

        p, r, f1 = prf(tp, fp, fn)
        per_class[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    for instance_id in instance_ids:
        if preds[instance_id]["verdict"] == golds[instance_id]["verdict"]:
            correct += 1

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(VERDICTS)

    return {
        "accuracy": correct / total if total else 0.0,
        "accuracy_denominator": total,
        "correct": correct,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def set_metrics(golds, preds, instance_ids, gold_key, pred_key):
    tp = fp = fn = 0

    for instance_id in instance_ids:
        gold_set = set(golds[instance_id].get(gold_key, []))
        pred_set = set(preds[instance_id].get(pred_key, []))

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    p, r, f1 = prf(tp, fp, fn)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "denominator_instances": len(instance_ids),
    }


def severe_false_warrant_rate(golds, preds, instance_ids):
    risky_total = 0
    severe_errors = 0

    for instance_id in instance_ids:
        if golds[instance_id]["verdict"] != "warranted":
            risky_total += 1
            if preds[instance_id]["verdict"] == "warranted":
                severe_errors += 1

    return {
        "rate": severe_errors / risky_total if risky_total else 0.0,
        "count": severe_errors,
        "denominator": risky_total,
    }


def abstention_by_gold_verdict(golds, preds):
    result = {}

    for label in VERDICTS:
        gold_ids = [instance_id for instance_id, gold in golds.items() if gold["verdict"] == label]
        abstained = [
            instance_id
            for instance_id in gold_ids
            if instance_id in preds and preds[instance_id].get("abstain", False)
        ]
        missing = [instance_id for instance_id in gold_ids if instance_id not in preds]

        result[label] = {
            "gold_count": len(gold_ids),
            "abstained_count": len(abstained),
            "missing_count": len(missing),
            "abstention_rate": len(abstained) / len(gold_ids) if gold_ids else 0.0,
        }

    return result


def rank_units(golds, preds):
    """Разложить все gold-инстансы в порядке убывания уверенности.

    Отвеченные идут по убыванию ``confidence``; инстансы с отказом ставятся в
    хвост как никогда не выбираемые добровольно. При покрытии, которое до них
    доходит, отказ считается ошибкой — вердикта система не дала, значит верным
    он быть не может, — но severe false warrant им не приписывается: ничего не
    было объявлено обоснованным.
    """
    answered, refused = [], []

    for instance_id, gold in golds.items():
        pred = preds.get(instance_id)
        risky = gold["verdict"] != "warranted"

        if pred is None or pred.get("abstain", False):
            refused.append({
                "instance_id": instance_id,
                "confidence": float("-inf"),
                "error": 1, "risky": int(risky), "severe": 0, "answered": 0,
            })
            continue

        error = int(pred["verdict"] != gold["verdict"])
        severe = int(risky and pred["verdict"] == "warranted")
        answered.append({
            "instance_id": instance_id,
            "confidence": float(pred["confidence"]),
            "error": error, "risky": int(risky), "severe": severe, "answered": 1,
        })

    answered.sort(key=lambda u: (-u["confidence"], u["instance_id"]))
    refused.sort(key=lambda u: u["instance_id"])
    return answered + refused


def _tie_groups(units):
    """Сгруппировать подряд идущие единицы с одинаковой уверенностью."""
    groups = []
    for unit in units:
        if groups and groups[-1][0] == unit["confidence"]:
            groups[-1][1].append(unit)
        else:
            groups.append((unit["confidence"], [unit]))
    return groups


def risk_coverage_curve(golds, preds):
    """Кривая risk-coverage по ранжированию. Точка на каждое k = 1..N.

    Внутри группы с равной уверенностью накопленные счётчики интерполируются
    линейно: это математическое ожидание при случайном разрешении ничьих,
    поэтому результат не зависит от того, в каком порядке лежат равные ответы.
    """
    units = rank_units(golds, preds)
    total = len(units)
    if not total:
        return []

    points = []
    base_err = base_sev = base_risky = 0.0
    consumed = 0

    for _, group in _tie_groups(units):
        size = len(group)
        sum_err = sum(u["error"] for u in group)
        sum_sev = sum(u["severe"] for u in group)
        sum_risky = sum(u["risky"] for u in group)

        for step in range(1, size + 1):
            fraction = step / size
            k = consumed + step
            err = base_err + fraction * sum_err
            sev = base_sev + fraction * sum_sev
            risky = base_risky + fraction * sum_risky

            points.append({
                "k": k,
                "coverage": k / total,
                "risk": err / k,
                "generalized_risk": err / total,
                "expected_errors": err,
                "severe_false_warrant_rate": (sev / risky) if risky > 0 else 0.0,
                "expected_severe_false_warrants": sev,
                "expected_risky": risky,
                "within_voluntary_coverage": k <= sum(u["answered"] for u in units),
            })

        base_err += sum_err
        base_sev += sum_sev
        base_risky += sum_risky
        consumed += size

    return points


def aurc(points):
    """Средний selective risk по всем уровням покрытия."""
    if not points:
        return 0.0
    return sum(p["risk"] for p in points) / len(points)


def augrc(points):
    """Средний обобщённый риск: ошибки нормируются на все N, а не на выбранные.

    AURC придаёт огромный вес области малого покрытия, где знаменатель крошечный
    и одна ошибка двигает метрику на десятки процентов. AUGRC от этого свободен.
    """
    if not points:
        return 0.0
    return sum(p["generalized_risk"] for p in points) / len(points)


def aurc_within_voluntary_coverage(points):
    """AURC, ограниченный тем покрытием, которое система выдала добровольно."""
    voluntary = [p for p in points if p["within_voluntary_coverage"]]
    if not voluntary:
        return 0.0
    return sum(p["risk"] for p in voluntary) / len(voluntary)


def metrics_at_fixed_coverage(points, levels=None):
    """Риск и SFWR при фиксированных уровнях покрытия.

    Для целевого покрытия c берётся префикс из ``ceil(c * N)`` ответов, то есть
    ближайшее достижимое покрытие не ниже c. Это делает системы сравнимыми в
    одной точке, чего кривая целиком не даёт.
    """
    levels = FIXED_COVERAGE_LEVELS if levels is None else levels
    if not points:
        return {str(level): None for level in levels}

    total = len(points)
    result = {}

    for level in levels:
        k = min(total, max(1, math.ceil(level * total)))
        point = points[k - 1]
        result[str(level)] = {
            "requested_coverage": level,
            "achieved_coverage": point["coverage"],
            "risk": point["risk"],
            "generalized_risk": point["generalized_risk"],
            "severe_false_warrant_rate": point["severe_false_warrant_rate"],
            "selected_count": point["k"],
            "within_voluntary_coverage": point["within_voluntary_coverage"],
        }

    return result


def coverage_at_target_sfwr(points):
    """Наибольшее покрытие, на котором SFWR не превышает цель.

    Рассматриваются только префиксы, которые система выдала добровольно: за
    границей добровольного покрытия в выборку попадают отказы, они не могут быть
    severe false warrant, и SFWR обманчиво падает.
    """
    result = {}
    voluntary = [p for p in points if p["within_voluntary_coverage"]]

    for target in TARGET_SFWR_LEVELS:
        good = [p["coverage"] for p in voluntary
                if p["severe_false_warrant_rate"] <= target]
        result[str(target)] = max(good, default=0.0)

    return result


def submission_report(golds, preds, missing, extra, duplicates, validation_errors):
    """Полнота сабмишена. Неполный не скорится — иначе молчание завышает метрики."""
    errors = []

    if missing:
        errors.append(
            f"нет предсказаний для {len(missing)} gold-инстанс(ов): "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}. "
            "Неуверенность выражается abstain=true, а не пропуском строки."
        )
    if extra:
        errors.append(
            f"{len(extra)} предсказан(ий) для инстансов вне gold: "
            f"{extra[:5]}{' …' if len(extra) > 5 else ''}"
        )
    if duplicates:
        errors.append(
            f"дубликаты предсказаний для {len(duplicates)} инстанс(ов): "
            f"{duplicates[:5]}{' …' if len(duplicates) > 5 else ''}"
        )
    if validation_errors:
        errors.append(f"{len(validation_errors)} предсказан(ий) не прошли валидацию")

    return {
        "complete": not errors,
        "scoreable": not errors,
        "expected_predictions": len(golds),
        "matched_predictions": len(set(golds) & set(preds)),
        "errors": errors,
    }


def make_markdown(metrics):
    counts = metrics["counts"]
    submission = metrics["submission"]

    lines = ["# Отчёт evaluator'а Track A", ""]

    if not submission["scoreable"]:
        lines += [
            "## СКОРИНГ ОТКЛОНЁН: сабмишен неполный",
            "",
            "Метрики качества не считались. Требуется ровно одно предсказание на",
            "каждый gold-инстанс: неуверенность выражается `abstain: true`, а не",
            "пропуском строки — иначе молчание на трудных инстансах завышало бы",
            "accuracy и F1.",
            "",
        ]
        for error in submission["errors"]:
            lines.append(f"- {error}")
        lines.append("")

    lines += [
        "## Сводка",
        "",
        f"- Инстансов в gold: {counts['gold_instances']}",
        f"- Предсказаний подано: {counts['predictions_submitted']}",
        f"- Валидных предсказаний: {counts['valid_predictions']}",
        f"- Отсутствующих предсказаний: {counts['missing_predictions']}",
        f"- Лишних предсказаний: {counts['extra_predictions']}",
        f"- Дубликатов: {counts['duplicate_predictions']}",
        f"- Невалидных предсказаний: {counts['invalid_predictions']}",
        f"- Предсказаний с отказом: {counts['abstained_predictions']}",
        f"- Предсказаний без отказа: {counts['non_abstained_predictions']}",
        "",
        "## Покрытие",
        "",
        f"- Coverage: {metrics['coverage']['value']:.4f} "
        f"({metrics['coverage']['answered']}/{metrics['coverage']['total']})",
        "",
    ]

    if not submission["scoreable"]:
        return "\n".join(lines) + "\n"

    lines += [
        "## Вердикт",
        "",
        f"- Accuracy: {metrics['verdict']['accuracy']:.4f}",
        f"- Знаменатель accuracy: {metrics['verdict']['accuracy_denominator']}",
        f"- Macro-F1: {metrics['verdict']['macro_f1']:.4f}",
        "",
        "Примечание: качество считается по отвеченным инстансам; инстансы с",
        "отказом видны в покрытии выше, а не растворяются в знаменателе.",
        "",
        "## Локализация evidence",
        "",
        f"- Precision: {metrics['evidence']['precision']:.4f}",
        f"- Recall: {metrics['evidence']['recall']:.4f}",
        f"- F1: {metrics['evidence']['f1']:.4f}",
        "",
        "## Issue-теги",
        "",
        f"- Precision: {metrics['issue_tags']['precision']:.4f}",
        f"- Recall: {metrics['issue_tags']['recall']:.4f}",
        f"- F1: {metrics['issue_tags']['f1']:.4f}",
        "",
        "## Отказ от ответа",
        "",
    ]

    for label, values in metrics["abstention_by_gold_verdict"].items():
        lines.append(
            f"- {label}: отказов {values['abstained_count']}/{values['gold_count']}, "
            f"доля={values['abstention_rate']:.4f}"
        )

    selective = metrics["selective_risk"]
    lines += [
        "",
        "## Selective risk",
        "",
        f"- AURC: {selective['aurc']:.4f}",
        f"- AUGRC: {selective['augrc']:.4f}",
        f"- AURC в пределах добровольного покрытия: "
        f"{selective['aurc_within_voluntary_coverage']:.4f}",
        f"- Добровольное покрытие: {selective['max_voluntary_coverage']:.4f}",
        "",
        "Кривая строится ранжированием ответов по уверенности, а не сеткой",
        "порогов: при равной уверенности у всех ответов пороговая сетка даёт",
        "одинаковое покрытие во всех точках и вырождает площадь.",
        "",
        "| Покрытие | Риск | Обобщённый риск | SFWR |",
        "|---|---|---|---|",
    ]

    for level, point in selective["at_fixed_coverage"].items():
        if point is None:
            continue
        lines.append(
            f"| {point['achieved_coverage']:.2f} | {point['risk']:.4f} | "
            f"{point['generalized_risk']:.4f} | "
            f"{point['severe_false_warrant_rate']:.4f} |"
        )

    sfwr = metrics["severe_false_warrant_rate_non_abstained"]
    lines += [
        "",
        "## Безопасность",
        "",
        f"- Severe false-warrant rate среди предсказаний без отказа: {sfwr['rate']:.4f}",
        f"- Severe false-warrant, число среди предсказаний без отказа: {sfwr['count']}",
    ]

    return "\n".join(lines) + "\n"


def score(pred_path, gold_path):
    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)

    golds = {}
    for row in gold_rows:
        golds[row["instance_id"]] = normalize_gold(row)

    preds = {}
    duplicates = []
    validation_errors = []

    for row in pred_rows:
        instance_id = row.get("instance_id", "<missing>")
        errors = validate_prediction(row)

        if errors:
            validation_errors.append({"instance_id": instance_id, "errors": errors})
            continue

        if instance_id in preds:
            duplicates.append(instance_id)
            validation_errors.append({"instance_id": instance_id,
                                      "errors": ["дубликат предсказания"]})
            continue

        preds[instance_id] = row

    missing = sorted(set(golds) - set(preds))
    extra = sorted(set(preds) - set(golds))
    duplicates = sorted(set(duplicates))

    non_abstained_ids = get_non_abstained_ids(golds, preds)
    abstained_count = sum(
        1 for instance_id in golds
        if instance_id in preds and preds[instance_id].get("abstain", False)
    )

    submission = submission_report(golds, preds, missing, extra, duplicates,
                                   validation_errors)
    total_gold = len(golds)

    metrics = {
        "submission": submission,
        "counts": {
            "gold_instances": total_gold,
            "predictions_submitted": len(pred_rows),
            "valid_predictions": len(preds),
            "invalid_predictions": len(validation_errors),
            "duplicate_predictions": len(duplicates),
            "missing_predictions": len(missing),
            "extra_predictions": len(extra),
            "abstained_predictions": abstained_count,
            "non_abstained_predictions": len(non_abstained_ids),
        },
        "coverage": {
            "answered": len(non_abstained_ids),
            "abstained": abstained_count,
            "total": total_gold,
            "value": len(non_abstained_ids) / total_gold if total_gold else 0.0,
        },
        "abstention_by_gold_verdict": abstention_by_gold_verdict(golds, preds),
        "missing_instance_ids": missing,
        "extra_instance_ids": extra,
        "duplicate_instance_ids": duplicates,
        "validation_errors": validation_errors,
    }

    # Неполный сабмишен не скорится. Иначе достаточно прислать несколько верных
    # ответов, чтобы получить высокую accuracy: пропущенные инстансы не попадали
    # бы в знаменатель. Диагностика выше остаётся, качественные метрики — null.
    if not submission["scoreable"]:
        metrics.update({
            "verdict": None,
            "evidence": None,
            "issue_tags": None,
            "severe_false_warrant_rate_non_abstained": None,
            "selective_risk": None,
        })
        return metrics

    curve = risk_coverage_curve(golds, preds)

    metrics.update({
        "verdict": verdict_metrics(golds, preds, non_abstained_ids),
        "evidence": set_metrics(golds, preds, non_abstained_ids,
                                "supporting_eids", "predicted_eids"),
        "issue_tags": set_metrics(golds, preds, non_abstained_ids,
                                  "issue_tags", "issue_tags"),
        "severe_false_warrant_rate_non_abstained":
            severe_false_warrant_rate(golds, preds, non_abstained_ids),
        "selective_risk": {
            "curve": curve,
            "aurc": aurc(curve),
            "augrc": augrc(curve),
            "aurc_within_voluntary_coverage": aurc_within_voluntary_coverage(curve),
            "max_voluntary_coverage":
                len(non_abstained_ids) / total_gold if total_gold else 0.0,
            "at_fixed_coverage": metrics_at_fixed_coverage(curve),
            "coverage_at_target_sfwr": coverage_at_target_sfwr(curve),
        },
    })

    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluator Track A для SciAudit-Stress.")
    parser.add_argument("--pred", required=True, help="Путь к JSONL с предсказаниями.")
    parser.add_argument("--gold", required=True, help="Путь к JSONL с приватным gold.")
    parser.add_argument("--out", default="metrics.json", help="Путь для записи метрик в JSON.")
    parser.add_argument("--report", default=None, help="Необязательный путь для отчёта в Markdown.")
    args = parser.parse_args(argv)

    metrics = score(args.pred, args.gold)

    Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    report = make_markdown(metrics)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")

    print(report)

    # Неполный сабмишен — отказ скоринга, а не низкая оценка. Код возврата должен
    # ронять CI, иначе неполный прогон молча попадёт в артефакты и в лидерборд.
    return 0 if metrics["submission"]["scoreable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
