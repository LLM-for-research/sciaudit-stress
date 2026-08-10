# Track A Evaluator

This evaluator compares system predictions with private gold labels.

## Run

Run the evaluator with:

    python -m sciaudit.evaluator.score \
      --pred examples/toy_predictions.jsonl \
      --gold examples/toy_gold.jsonl \
      --out metrics.json \
      --report report.md

## Metrics

### Verdict quality

- Verdict accuracy
- Verdict macro-F1
- Per-class precision / recall / F1

Abstained instances are excluded from the verdict accuracy denominator. Accuracy is computed only over valid, non-abstained predictions.

### Evidence localization

- Evidence ID precision
- Evidence ID recall
- Evidence ID F1

### Issue tags

- Issue tag precision
- Issue tag recall
- Issue tag F1

### Safety

- Severe false-warrant rate among non-abstained predictions

A severe false warrant happens when the gold verdict is `overclaimed`, `contradicted`, or `insufficient`, but the system predicts `warranted`.

### Abstention and selective risk

The evaluator reports:

- abstention count
- non-abstained prediction count
- abstention rate by gold verdict
- coverage at multiple confidence thresholds
- risk at multiple confidence thresholds
- AURC
- coverage at target SFWR levels

Coverage is the fraction of gold instances selected for scoring at a confidence threshold.

Risk is the error rate among selected non-abstained predictions.

## Outputs

- `metrics.json`: machine-readable metrics
- `report.md`: readable Markdown summary
