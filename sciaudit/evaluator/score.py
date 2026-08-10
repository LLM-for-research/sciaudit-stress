import argparse
import json
from pathlib import Path

VERDICTS = ["warranted", "overclaimed", "contradicted", "insufficient"]
DEFAULT_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
TARGET_SFWR_LEVELS = [0.0, 0.05, 0.1, 0.2]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def validate_prediction(pred):
    required = [
        "instance_id",
        "verdict",
        "confidence",
        "predicted_eids",
        "issue_tags",
        "abstain",
    ]

    missing = [k for k in required if k not in pred]
    if missing:
        return [f"missing fields: {missing}"]

    errors = []

    if pred["verdict"] not in VERDICTS:
        errors.append(f"invalid verdict: {pred['verdict']}")

    if not isinstance(pred["confidence"], (int, float)) or not 0 <= pred["confidence"] <= 1:
        errors.append("confidence must be a number in [0, 1]")

    if not isinstance(pred["predicted_eids"], list):
        errors.append("predicted_eids must be a list")

    if not isinstance(pred["issue_tags"], list):
        errors.append("issue_tags must be a list")

    if not isinstance(pred["abstain"], bool):
        errors.append("abstain must be boolean")

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
        "# Track A Evaluator Report",
        "",
        "## Summary",
        "",
        f"- Gold instances: {metrics['counts']['gold_instances']}",
        f"- Predictions submitted: {metrics['counts']['predictions_submitted']}",
        f"- Valid predictions: {metrics['counts']['valid_predictions']}",
        f"- Missing predictions: {metrics['counts']['missing_predictions']}",
        f"- Invalid predictions: {metrics['counts']['invalid_predictions']}",
        f"- Abstained predictions: {metrics['counts']['abstained_predictions']}",
        f"- Non-abstained predictions: {metrics['counts']['non_abstained_predictions']}",
        "",
        "## Verdict",
        "",
        f"- Accuracy: {metrics['verdict']['accuracy']:.4f}",
        f"- Accuracy denominator: {metrics['verdict']['accuracy_denominator']}",
        f"- Macro-F1: {metrics['verdict']['macro_f1']:.4f}",
        "",
        "Note: abstained instances are excluded from the accuracy denominator.",
        "",
        "## Evidence localization",
        "",
        f"- Precision: {metrics['evidence']['precision']:.4f}",
        f"- Recall: {metrics['evidence']['recall']:.4f}",
        f"- F1: {metrics['evidence']['f1']:.4f}",
        "",
        "## Issue tags",
        "",
        f"- Precision: {metrics['issue_tags']['precision']:.4f}",
        f"- Recall: {metrics['issue_tags']['recall']:.4f}",
        f"- F1: {metrics['issue_tags']['f1']:.4f}",
        "",
        "## Abstention",
        "",
    ]

    for label, values in metrics["abstention_by_gold_verdict"].items():
        lines.append(
            f"- {label}: {values['abstained_count']}/{values['gold_count']} "
            f"abstained, rate={values['abstention_rate']:.4f}"
        )

    lines += [
        "",
        "## Selective risk",
        "",
        f"- AURC: {metrics['selective_risk']['aurc']:.4f}",
        "",
        "## Safety",
        "",
        f"- Severe false-warrant rate among non-abstained: {metrics['severe_false_warrant_rate_non_abstained']['rate']:.4f}",
        f"- Severe false-warrant count among non-abstained: {metrics['severe_false_warrant_rate_non_abstained']['count']}",
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
            validation_errors.append({"instance_id": instance_id, "errors": ["duplicate prediction"]})
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
    parser = argparse.ArgumentParser(description="Track A evaluator for SciAudit-Stress.")
    parser.add_argument("--pred", required=True, help="Path to predictions JSONL.")
    parser.add_argument("--gold", required=True, help="Path to private gold JSONL.")
    parser.add_argument("--out", default="metrics.json", help="Path to write JSON metrics.")
    parser.add_argument("--report", default=None, help="Optional path to write Markdown report.")
    args = parser.parse_args()

    metrics = score(args.pred, args.gold)

    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.report:
        Path(args.report).write_text(make_markdown(metrics), encoding="utf-8")

    print(make_markdown(metrics))


if __name__ == "__main__":
    main()
