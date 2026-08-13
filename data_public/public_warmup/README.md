# Public warm-up slice (Track A)

First runnable public dataset for the readiness loop. Everything here is
**synthetic placeholder content** — see [manifest.json](manifest.json) for
per-instance provenance notes and the replacement policy once the controlled
paper pool (Task 2) lands.

| File | Contents |
|---|---|
| `inputs.jsonl` | 17 Track A input instances (`track_a_input_v1` schema) |
| `gold.jsonl` | 17 gold records, one per `instance_id` (format of `examples/toy_gold.jsonl`) |
| `manifest.json` | Staff-side provenance: synthetic markers, neutral origins, dataset shape (never merged into inputs) |

Properties guaranteed by `tests/test_public_warmup.py`:

- all four verdicts are present and none exceeds 50%;
- non-semantic `sas_` instance IDs — the verdict cannot be guessed from metadata;
- no private fields in `inputs.jsonl` — enforced by the Track A input validator
  and the leakage tests;
- the same evidence pack can support claims that are judged differently, so
  the verdict is not readable from the evidence alone;
- B0 runs on `inputs.jsonl` and the evaluator produces metrics without errors.

## Local verification

```bash
uv run python -m sciaudit.schemas.validate_inputs data_public/public_warmup/inputs.jsonl

uv run python -m sciaudit.baselines.b0_always_insufficient \
  --input data_public/public_warmup/inputs.jsonl --output /tmp/b0_warmup.jsonl

uv run python -m sciaudit.schemas.validate_predictions /tmp/b0_warmup.jsonl \
  --input data_public/public_warmup/inputs.jsonl

uv run python -m sciaudit.evaluator.score \
  --pred /tmp/b0_warmup.jsonl --gold data_public/public_warmup/gold.jsonl \
  --out /tmp/warmup_metrics.json
```
