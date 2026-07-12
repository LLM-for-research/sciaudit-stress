import argparse
import json
from pathlib import Path
from collections import defaultdict

VERDICTS = ["warranted", "overclaimed", "contradicted", "insufficient"]


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


def verdict_metrics(golds, preds):
    correct = 0
    total = len(golds)
    per_class = {}

    for label in VERDICTS:
        tp = fp = fn = 0
        for instance_id, gold in golds.items():
            gold_label = gold["verdict"]
            pred_label = preds.get(instance_id, {}).get("verdict")

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

    for instance_id, gold in golds.items():
        pred = preds.get(instance_id)
        if pred and pred["verdict"] == gold["verdict"]:
            correct += 1

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(VERDICTS)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def set_metrics(golds, preds, gold_key, pred_key):
    tp = fp = fn = 0

    for instance_id, gold in golds.items():
        gold_set = set(gold.get(gold_key, []))
        pred_set = set(preds.get(instance_id, {}).get(pred_key, []))

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
    }


def severe_false_warrant_rate(golds, preds):
    risky_total = 0
    severe_errors = 0

    for instance_id, gold in golds.items():
        if gold["verdict"] != "warranted":
            risky_total += 1
            pred = preds.get(instance_id)
            if pred and pred["verdict"] == "warranted":
                severe_errors += 1

    return {
        "rate": severe_errors / risky_total if risky_total else 0.0,
        "count": severe_errors,
        "denominator": risky_total,
    }


def normalize_gold(row):
    gold = row.get("gold", row)
    return {
        "verdict": gold["verdict"],
        "supporting_eids": gold.get("supporting_eids", []),
        "issue_tags": gold.get("issue_tags", []),
    }


def make_markdown(metrics):
    lines = [
        "# Track A Evaluator Report",
        "",
        "## Summary",
        "",
        f"- Gold instances: {metrics['counts']['gold_instances']}",
        f"- Predictions submitted: {metrics['counts']['predictions_submitted']}",
        f"- Missing predictions: {metrics['counts']['missing_predictions']}",
        f"- Invalid predictions: {metrics['counts']['invalid_predictions']}",
        "",
        "## Verdict",
        "",
        f"- Accuracy: {metrics['verdict']['accuracy']:.4f}",
        f"- Macro-F1: {metrics['verdict']['macro_f1']:.4f}",
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
        "## Safety",
        "",
        f"- Severe false-warrant rate: {metrics['severe_false_warrant_rate']['rate']:.4f}",
        f"- Severe false-warrant count: {metrics['severe_false_warrant_rate']['count']}",
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

    metrics = {
        "counts": {
            "gold_instances": len(golds),
            "predictions_submitted": len(pred_rows),
            "valid_predictions": len(preds),
            "invalid_predictions": len(validation_errors),
            "missing_predictions": len(missing),
            "extra_predictions": len(extra),
        },
        "verdict": verdict_metrics(golds, preds),
        "evidence": set_metrics(golds, preds, "supporting_eids", "predicted_eids"),
        "issue_tags": set_metrics(golds, preds, "issue_tags", "issue_tags"),
        "severe_false_warrant_rate": severe_false_warrant_rate(golds, preds),
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
