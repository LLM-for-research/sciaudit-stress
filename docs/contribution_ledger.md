# Contribution ledger

A running record of **who contributed what**, with a link to the artifact and its review status.
It exists so contributions are visible and creditable, and so authorship decisions later are based
on documented work rather than memory. **Authorship is not automatic** — it follows from
substantial, documented, verified contributions recorded here.

Add a row when your work is merged (or when an artifact is accepted). Keep it in this file, or in a
linked sheet if the class prefers — the columns are the contract.

| Date | Contributor | Contribution (what) | Type | Artifact link | Review status | Verifier | CRediT-style role |
|---|---|---|---|---|---|---|---|
| 2026-07-12 | Rodion Krainov | Implemented B0 baseline module | code | #3 / [commit](https://github.com/LLM-for-research/sciaudit-stress/commit/5344fa852e12fade7191797d9ed2cfc554c49ac1) | merged | - | Software |
| 2026-07-12 | Abdullo Muminov | Implemented B0 baseline basic evaluator and leaderboard report | code | #5 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/7) | merged | Rodion Krainov | Software |
| | | | | | | | |

## Column meanings

- **Contribution (what)** — one concrete line; not "helped with stuff."
- **Type** — one of: `data`, `stress-case`, `code`, `evaluation`, `docs`, `validation`,
  `analysis`, `writing`, `release`.
- **Artifact link** — a PR number, file path, or experiment ID. **Required** — an unlinked
  contribution can't be credited.
- **Review status** — `proposed` / `in-review` / `merged` / `rejected`.
- **Verifier** — the TA or peer who signed off (ties to the two-sign-off rule in the
  [Week-1 guide](week1_guide.md)).
- **CRediT-style role** — e.g. Software, Data curation, Validation, Analysis, Writing, Visualization.

## How this connects to the rest

- Each row should trace back to a PR that carried a [weekly update](weekly_update_template.md).
- Real systems referenced here should have a [system card](system_card_template.md).
- Useful-but-not-authorship-level work is still recorded and **acknowledged** — record everything;
  credit is decided later, transparently, from these rows.
