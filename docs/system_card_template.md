# System-card template

Every real system (baseline or team) ships a system card. It tells a reader what the system is,
what it may and may not be used for, and what it costs — before they look at any leaderboard number.
Copy the block, fill it in, and keep it next to your system's code.

```markdown
# System card: <system name>

**System name:**            <!-- e.g. team04-bm25-audit-v2 -->
**Team / author:**
**Version:**
**Track:**                  <!-- A / B / C / D -->

## Inputs
**Input used:**             <!-- which split(s): public_warmup, public_dev, hidden_val, ... -->
**Evidence handling:**      <!-- full pack / retrieved top-k / numeric-checked / ... -->

## Components
**Tools used:**             <!-- BM25, numeric checker, VLM, none, ... -->
**Model used:**             <!-- approved open model id, or "none" -->
**Uses numeric checker?**   yes / no
**Uses fine-tuning?**       yes / no
**Uses VLM?**               yes / no

## Behavior & limits
**Intended use:**           <!-- what this system is for -->
**Non-intended use:**       <!-- what it must NOT be used to claim -->
**Limitations:**            <!-- known weaknesses, failure modes -->
**Abstention behavior:**    <!-- when it abstains, if at all -->
**Severe false-warrant behavior:**  <!-- how often it wrongly says "warranted" -->

## Resources
**Cost / runtime:**         <!-- runtime per instance; gpu_seconds; api_cost_usd (0 for open) -->

## Reproducibility & responsible use
**How to run:**             <!-- exact command(s) -->
**Responsible LLM-use summary:**  <!-- what an LLM helped build; what was verified by hand -->
```

## Notes

- **Input used / tools used / model used / limitations / cost-runtime** are the required minimum.
  The extra fields (intended/non-intended use, abstention, severe false-warrant, reproducibility)
  are what turn a card from a label into something a reviewer can trust — fill them when they apply.
- Keep the card honest about **cost/runtime**. Evaluated systems use approved open models, so
  `api_cost_usd` is normally `0`; report `gpu_seconds` and wall-clock runtime truthfully.
- The card must match the machine-readable `system_info` block your predictions emit (model,
  `uses_numeric_checker`, `uses_lora`, `uses_vlm`). If they disagree, fix the card.
- Responsible-use summary follows the [LLM-use policy](llm_use_policy.md).
