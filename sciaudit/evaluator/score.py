"""Evaluator Track A: сравнение предсказаний с приватным gold (мануал §12).

Считает качество вердикта, локализацию evidence, issue-теги, severe false-warrant
rate (§3.4) и selective risk (§3.5, §12.5). Пишет метрики в JSON и читаемый
отчёт в Markdown.

Запуск:
    python -m sciaudit.evaluator.score --pred preds.jsonl --gold gold.jsonl \\
        --out metrics.json --report report.md
"""
import argparse
import json
from pathlib import Path

from sciaudit.schemas import read_jsonl as _read_jsonl_with_line_numbers

VERDICTS = ["warranted", "overclaimed", "contradicted", "insufficient"]
DEFAULT_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
TARGET_SFWR_LEVELS = [0.0, 0.05, 0.1, 0.2]


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


def selected_ids_at_threshold(golds, preds, threshold):
    return [
        instance_id
        for instance_id in golds
        if instance_id in preds
        and not preds[instance_id].get("abstain", False)
        and preds[instance_id]["confidence"] >= threshold
    ]


def risk_coverage_curve(golds, preds, thresholds=None):
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    points = []

    for threshold in thresholds:
        selected = selected_ids_at_threshold(golds, preds, threshold)
        selected_count = len(selected)
        errors = sum(1 for instance_id in selected if preds[instance_id]["verdict"] != golds[instance_id]["verdict"])
        coverage = selected_count / len(golds) if golds else 0.0
        risk = errors / selected_count if selected_count else 0.0
        sfwr = severe_false_warrant_rate(golds, preds, selected)

        points.append({
            "threshold": threshold,
            "coverage": coverage,
            "risk": risk,
            "selected_count": selected_count,
            "error_count": errors,
            "severe_false_warrant_rate": sfwr["rate"],
            "severe_false_warrant_count": sfwr["count"],
            "severe_false_warrant_denominator": sfwr["denominator"],
        })

    return points


def aurc(points):
    if not points:
        return 0.0

    sorted_points = sorted(points, key=lambda p: p["coverage"])
    area = 0.0

    for left, right in zip(sorted_points, sorted_points[1:]):
        width = right["coverage"] - left["coverage"]
        height = (left["risk"] + right["risk"]) / 2
        area += width * height

    return area


def coverage_at_target_sfwr(points):
    result = {}

    for target in TARGET_SFWR_LEVELS:
        valid_points = [
            point for point in points
            if point["severe_false_warrant_rate"] <= target
        ]
        best_coverage = max((point["coverage"] for point in valid_points), default=0.0)
        result[str(target)] = best_coverage

    return result


def make_markdown(metrics):
    lines = [
        "# Отчёт evaluator'а Track A",
        "",
        "## Сводка",
        "",
        f"- Инстансов в gold: {metrics['counts']['gold_instances']}",
        f"- Предсказаний подано: {metrics['counts']['predictions_submitted']}",
        f"- Валидных предсказаний: {metrics['counts']['valid_predictions']}",
        f"- Отсутствующих предсказаний: {metrics['counts']['missing_predictions']}",
        f"- Невалидных предсказаний: {metrics['counts']['invalid_predictions']}",
        f"- Предсказаний с отказом: {metrics['counts']['abstained_predictions']}",
        f"- Предсказаний без отказа: {metrics['counts']['non_abstained_predictions']}",
        "",
        "## Вердикт",
        "",
        f"- Accuracy: {metrics['verdict']['accuracy']:.4f}",
        f"- Знаменатель accuracy: {metrics['verdict']['accuracy_denominator']}",
        f"- Macro-F1: {metrics['verdict']['macro_f1']:.4f}",
        "",
        "Примечание: инстансы с отказом исключены из знаменателя accuracy.",
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

    lines += [
        "",
        "## Selective risk",
        "",
        f"- AURC: {metrics['selective_risk']['aurc']:.4f}",
        "",
        "## Безопасность",
        "",
        f"- Severe false-warrant rate среди предсказаний без отказа: {metrics['severe_false_warrant_rate_non_abstained']['rate']:.4f}",
        f"- Severe false-warrant, число среди предсказаний без отказа: {metrics['severe_false_warrant_rate_non_abstained']['count']}",
    ]

    return "\n".join(lines) + "\n"


def score(pred_path, gold_path):
    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)

    golds = {}
    for row in gold_rows:
        golds[row["instance_id"]] = normalize_gold(row)

    preds = {}
    validation_errors = []

    for row in pred_rows:
        instance_id = row.get("instance_id", "<missing>")
        errors = validate_prediction(row)

        if errors:
            validation_errors.append({"instance_id": instance_id, "errors": errors})
            continue

        if instance_id in preds:
            validation_errors.append({"instance_id": instance_id, "errors": ["дубликат предсказания"]})
            continue

        preds[instance_id] = row

    missing = sorted(set(golds) - set(preds))
    extra = sorted(set(preds) - set(golds))

    non_abstained_ids = get_non_abstained_ids(golds, preds)
    abstained_count = sum(
        1 for instance_id in golds
        if instance_id in preds and preds[instance_id].get("abstain", False)
    )

    curve = risk_coverage_curve(golds, preds)

    metrics = {
        "counts": {
            "gold_instances": len(golds),
            "predictions_submitted": len(pred_rows),
            "valid_predictions": len(preds),
            "invalid_predictions": len(validation_errors),
            "missing_predictions": len(missing),
            "extra_predictions": len(extra),
            "abstained_predictions": abstained_count,
            "non_abstained_predictions": len(non_abstained_ids),
        },
        "verdict": verdict_metrics(golds, preds, non_abstained_ids),
        "evidence": set_metrics(golds, preds, non_abstained_ids, "supporting_eids", "predicted_eids"),
        "issue_tags": set_metrics(golds, preds, non_abstained_ids, "issue_tags", "issue_tags"),
        "severe_false_warrant_rate_non_abstained": severe_false_warrant_rate(golds, preds, non_abstained_ids),
        "abstention_by_gold_verdict": abstention_by_gold_verdict(golds, preds),
        "selective_risk": {
            "thresholds": curve,
            "aurc": aurc(curve),
            "coverage_at_target_sfwr": coverage_at_target_sfwr(curve),
        },
        "missing_instance_ids": missing,
        "extra_instance_ids": extra,
        "validation_errors": validation_errors,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluator Track A для SciAudit-Stress.")
    parser.add_argument("--pred", required=True, help="Путь к JSONL с предсказаниями.")
    parser.add_argument("--gold", required=True, help="Путь к JSONL с приватным gold.")
    parser.add_argument("--out", default="metrics.json", help="Путь для записи метрик в JSON.")
    parser.add_argument("--report", default=None, help="Необязательный путь для отчёта в Markdown.")
    args = parser.parse_args()

    metrics = score(args.pred, args.gold)

    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.report:
        Path(args.report).write_text(make_markdown(metrics), encoding="utf-8")

    print(make_markdown(metrics))


if __name__ == "__main__":
    main()
