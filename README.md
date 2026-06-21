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