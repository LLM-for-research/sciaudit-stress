# Data schemas: public input, prediction, private gold

Track A uses a **hard three-layer separation**. A system only ever sees layer 1
and only ever produces layer 2. Layer 3 exists solely in the private
`sciaudit-stress-private` repository.

| Layer | Schema | Who sees it | Contains |
|---|---|---|---|
| 1. Public input | [`schemas/track_a_input.schema.json`](../schemas/track_a_input.schema.json) | students + systems | abstracted `paper_id`, `instance_id`, normalized claim, frozen evidence pack, `allowed_evidence_ids` |
| 2. Prediction | [`schemas/prediction.schema.json`](../schemas/prediction.schema.json) | produced by systems, consumed by evaluator | verdict, evidence IDs, issue tags, confidence, optional abstention + runtime/cost |
| 3. Internal annotation (private gold) | [`schemas/internal_annotation.schema.json`](../schemas/internal_annotation.schema.json) *(draft)* | staff/TA only | everything above **plus** provenance, stress metadata, gold verdict, severity, review notes, private rationale, split |

Why the separation is strict: stress metadata trivially leaks labels (seeing
`stress_type=evidence_removal` lets a system guess `insufficient` for free),
and gold labels/provenance must never reach systems or external LLMs. See
[benchmark framing](benchmark_framing.md).

---

## 1. Public input (`track_a_input.schema.json`)

One JSONL line per instance. `additionalProperties: false` everywhere — any
extra key (e.g. a leaked `gold` block) makes the file invalid.

```json
{
  "schema_version": "track_a_input_v1",
  "paper_id": "P001",
  "instance_id": "sas_8f3kq2m9",
  "claim": {
    "text": "The method consistently outperforms the compared baselines across all reported robustness benchmarks.",
    "claim_type": "baseline_superiority",
    "scope": "Judge only from the supplied evidence pack."
  },
  "evidence_pack": [
    {"eid": "e01", "source_kind": "table", "modality": "table_text", "text": "..."},
    {"eid": "e02", "source_kind": "caption", "modality": "caption_text", "text": "..."}
  ],
  "allowed_evidence_ids": ["e01", "e02"]
}
```

- `paper_id` is **abstracted** (no titles/authors/venues/URLs); the mapping to
  real papers lives in the private provenance map.
- `allowed_evidence_ids` are the only IDs a prediction may cite; the validator
  checks they are a subset of the pack's `eid`s.
- `claim_type` is one of the documented claim types (see
  [`configs/allowed_labels.yaml`](../configs/allowed_labels.yaml)):
  `numerical_performance`, `baseline_superiority`, `ablation`, `robustness`,
  `efficiency`, `bounded_generalization`.
- **No private labels or metadata, ever.** The validator explicitly rejects
  `gold`, `verdict`, `severity`, `stress`/`stress_type`, `private_rationale`,
  `provenance`/`provenance_map`, `review_note`, `split`, and similar keys at
  any nesting depth.

Examples: [`examples/sample_inputs.jsonl`](../examples/sample_inputs.jsonl)
(6 instances covering all claim types).

## 2. Prediction (`prediction.schema.json`)

One JSONL line per input line, `instance_id` preserved.

**Required:** `instance_id`, `verdict`, `confidence`, `predicted_eids`,
`issue_tags`.
**Optional (recommended; baselines always emit them):** `abstain`,
`rationale_short`, `runtime_seconds`, `estimated_cost`, `system_info`.

- **Verdicts are fixed:** `warranted` | `overclaimed` | `contradicted` |
  `insufficient`. Nothing else validates.
- **Issue tags are fixed** to the documented list in
  [`configs/allowed_labels.yaml`](../configs/allowed_labels.yaml):
  `numerical_inconsistency`, `claim_stronger_than_evidence`,
  `unsupported_generalization`, `missing_ablation_support`,
  `non_comparable_baseline`, `weak_statistical_support`,
  `evidence_missing_or_incomplete`, `caption_chart_mismatch`.
- `predicted_eids` (the "evidence_ids" of the prediction) must come from the
  instance's `allowed_evidence_ids`; the validator cross-checks this when given
  `--input`.
- `confidence` ∈ [0, 1]. The gold label `insufficient` and the system action
  `abstain` remain distinct concepts.

Examples: [`examples/sample_predictions.jsonl`](../examples/sample_predictions.jsonl)
(6 predictions: all four verdicts, one abstention, one minimal
required-fields-only object).

## 3. Internal annotation (`internal_annotation.schema.json`, draft)

The private, staff/TA-only record from which public inputs are derived by
**stripping** provenance, stress metadata, gold, review fields, and split. It
adds:

- `paper.provenance_ref` — key into the private provenance map;
- `evidence_pack[].source_ref`, `is_distractor` — private pointers/markers;
- `stress` — `is_stress_case`, `stress_type` (10 documented transformations),
  `seed_instance_id`;
- `gold` — verdict, `supporting_eids`, issue tags, `severity`
  (`minor`/`moderate`/`severe`), `private_rationale`;
- `review` — `validation_level` (`auto_unchecked` → `team_verified` →
  `ta_validated` → `ta_adjudicated`), reviewer, `review_note`,
  `human_verification_note`;
- `split` — benchmark tier.

**Real internal annotations must never be committed to this repository.**
[`examples/sample_internal_annotation.synthetic.jsonl`](../examples/sample_internal_annotation.synthetic.jsonl)
contains two clearly marked *synthetic* records for schema documentation only.

---

## Validators

```bash
# public inputs: schema + leakage + eid consistency
uv run python -m sciaudit.schemas.validate_inputs examples/sample_inputs.jsonl

# predictions: schema + duplicates (+ coverage/eid cross-check with --input)
uv run python -m sciaudit.schemas.validate_predictions examples/sample_predictions.jsonl \
  --input examples/sample_inputs.jsonl
```

Exit code 0 = valid; 1 = errors (printed with `file:line` locations). Both run
in CI. Tests: `tests/test_schema_validation.py`.

`configs/allowed_labels.yaml` is the single source of truth for verdicts,
issue tags, and claim types; a test asserts the JSON Schema enums stay in sync
with it.
