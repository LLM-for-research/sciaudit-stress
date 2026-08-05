# Public warm-up slice (Track A)

First runnable public dataset for the readiness loop. Everything here is
**synthetic placeholder content** — see [manifest.json](manifest.json) for the
per-instance provenance note and the replacement policy once the controlled
paper pool (Task 2) lands.

| File | Contents |
|---|---|
| `inputs.jsonl` | 15 Track A input instances (`track_a_input_v1` schema) |
| `gold.jsonl` | 15 gold records, one per `instance_id` (format of `examples/toy_gold.jsonl`) |
| `manifest.json` | Staff-side provenance: synthetic markers, origins, verdict balance (never merged into inputs) |

Properties guaranteed by `tests/test_public_warmup.py`:

- all four verdicts present (`warranted` 3, `overclaimed` 6, `contradicted` 3,
  `insufficient` 3); no verdict exceeds 50%;
- non-semantic `sas_` instance IDs — the verdict cannot be guessed from metadata;
- no private fields (gold, stress, provenance, severity, private rationale);
- the same evidence pack can support different claims (e.g.
  `sas_j3n8c4k9` = warranted vs `sas_l6v9k2b8` = overclaimed) so that the
  verdict is not readable from the evidence alone;
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
