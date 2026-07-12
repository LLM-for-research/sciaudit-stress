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

- Verdict accuracy
- Verdict macro-F1
- Per-class precision / recall / F1
- Evidence ID precision / recall / F1
- Issue tag precision / recall / F1
- Severe false-warrant rate

A severe false warrant happens when the gold verdict is `overclaimed`, `contradicted`, or `insufficient`, but the system predicts `warranted`.

## Outputs

- `metrics.json`: machine-readable metrics
- `report.md`: readable Markdown summary
