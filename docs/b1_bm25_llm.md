# B1 BM25 + structured model audit baseline

B1 is the first real baseline after B0. It retrieves the most relevant evidence units from the supplied evidence pack using BM25 and asks a structured model prompt to produce a prediction.

## Run

    python -m sciaudit.baselines.b1_bm25_llm \
      --input examples/sample_inputs.jsonl \
      --output predictions.jsonl \
      --top-k 3

Optional model command:

    python -m sciaudit.baselines.b1_bm25_llm \
      --input examples/sample_inputs.jsonl \
      --output predictions.jsonl \
      --top-k 3 \
      --model-command "your-open-model-command"

The prompt is passed to the model command through stdin.

## Behavior

- Tokenizes claim and evidence units.
- Ranks evidence units with BM25.
- Builds a strict structured JSON prompt.
- Parses model JSON.
- Retries invalid JSON.
- Uses a safe fallback if parsing still fails.
- Filters `predicted_eids` so they are always a subset of allowed evidence IDs.
- Writes schema-compatible predictions JSONL.

## Safe fallback

If the model returns invalid JSON after retries, B1 returns:

- `verdict`: `insufficient`
- `confidence`: `0.0`
- `predicted_eids`: `[]`
- `issue_tags`: `["model_parse_failure"]`
- `abstain`: `true`
