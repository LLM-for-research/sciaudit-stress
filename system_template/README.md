# System template

Fork/copy this directory to build a SciAudit-Stress system. Keep the run
contract identical; change only what's *inside* `audit_instance`.

## Interface (do not change)

```bash
python run.py --input inputs.jsonl --output predictions.jsonl
```

- `--input`: JSONL of Track A input objects (`../schemas/track_a_input.schema.json`)
- `--output`: JSONL of prediction objects (`../schemas/prediction.schema.json`)
- one prediction line per input line, `instance_id` preserved.

## Docker

```bash
docker build -t my-team-system .
docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output \
  my-team-system python run.py --input /input/inputs.jsonl --output /output/predictions.jsonl
```

## Dependencies (uv)

```bash
uv add transformers==4.44.2 rank-bm25==0.2.2   # example pinned deps
uv lock                                         # refresh uv.lock (commit it)
uv run python run.py --input examples/in.jsonl --output preds.jsonl
```

## What to edit

Replace the `audit_instance` function in `run.py` with your pipeline (retrieval,
LLM call, numeric checker, abstention, …). Add deps with `uv add` (versions are
pinned in `uv.lock` — commit it). Evaluated systems must use approved open models
only — no paid/hidden external APIs (staff manual §7.4).
