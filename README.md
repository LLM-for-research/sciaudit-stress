# SciAudit-Stress — central public repository

Public rail for the SciAudit-Stress benchmark-and-systems course project:
schemas, public data, baselines, evaluator, leakage tools, the reusable system
template, docs, and tests.

Built **rails-before-trains** (staff manual §1.3): the pipeline runs end-to-end
with a placeholder system *before* any real model is added.

> **Private data lives elsewhere.** Hidden inputs, gold labels, stress metadata,
> provenance, and `private_eval.schema.json` belong in the separate, access-limited
> `sciaudit-stress-private` repository — never here. The `.gitignore` in this repo
> is a safety net, not the primary control.

## Run contract

Every system — baseline or team — obeys one interface:

```bash
python system_template/run.py --input inputs.jsonl --output predictions.jsonl
```

or containerized:

```bash
docker build -t sciaudit-system system_template/
docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output \
  sciaudit-system python run.py --input /input/inputs.jsonl --output /output/predictions.jsonl
```

Whatever lives inside the system (BM25, LLM, numeric checker, abstention, LoRA),
the external contract is identical.

## Data contract

Three layers, strictly separated (see [docs/schemas.md](docs/schemas.md)):
public Track A input (`schemas/track_a_input.schema.json`), prediction
(`schemas/prediction.schema.json`), and the private internal annotation draft
(`schemas/internal_annotation.schema.json`). Validate files with:

```bash
uv run python -m sciaudit.schemas.validate_inputs examples/sample_inputs.jsonl
uv run python -m sciaudit.schemas.validate_predictions examples/sample_predictions.jsonl \
  --input examples/sample_inputs.jsonl
```

## Readiness loop (public warm-up)

The full chain CI runs on the public warm-up slice (staff manual §8.1),
runnable locally in this order:

```bash
# 1. public inputs are schema-valid and leakage-free
uv run python -m sciaudit.schemas.validate_inputs data_public/public_warmup/inputs.jsonl

# 2. a system runs over the inputs (B0 = trivial always-insufficient)
uv run python -m sciaudit.baselines.b0_always_insufficient \
  --input data_public/public_warmup/inputs.jsonl --output /tmp/b0_warmup.jsonl

# 3. the system's predictions are schema-valid and cite allowed evidence
uv run python -m sciaudit.schemas.validate_predictions /tmp/b0_warmup.jsonl \
  --input data_public/public_warmup/inputs.jsonl

# 4. no private fields anywhere a student could read them
uv run python -m sciaudit.leakage.forbidden_key_scan data_public/ examples/

# 5. score predictions against the gold labels
uv run python -m sciaudit.evaluator.score \
  --pred /tmp/b0_warmup.jsonl --gold data_public/public_warmup/gold.jsonl \
  --out /tmp/warmup_metrics.json
```