# Week-1 guide

Welcome. This guide gets you from *nothing* to *first merged contribution* without needing a
meeting. If something here is unclear, that's a documentation bug — open an issue.

**Context first (5 min):** skim [What this is / what this is not](benchmark_framing.md). Key point:
**Sprint 1 is Track A — the minimal controlled slice, not the final benchmark.** You are helping
build a rail, then stress it.

---

## 1. What to do first

Prerequisites: `git`, Python ≥ 3.11, and [uv](https://docs.astral.sh/uv/) (our package manager —
we do **not** use `pip` directly).

```bash
# 1. get the repo
git clone <central-repo-url> sciaudit-stress
cd sciaudit-stress

# 2. create the environment + install everything from the lockfile
uv sync

# 3. prove the rail runs end-to-end
uv run pytest -q
uv run python system_template/run.py --input examples/sample_inputs.jsonl --output /tmp/preds.jsonl
uv run python -m sciaudit.baselines.b0_always_insufficient \
  --input examples/sample_inputs.jsonl --output /tmp/b0.jsonl
```

If `pytest` is green and both commands write a `predictions` file, your environment is correct.
Now read three short files so you understand the contract:

- `schemas/student_input.schema.json` — the only thing a system receives.
- `schemas/prediction.schema.json` — the only thing a system may output.
- `sciaudit/baselines/b0_always_insufficient.py` — the smallest complete system (B0).

**Golden rule:** every system, baseline or yours, obeys the same interface —
`--input inputs.jsonl --output predictions.jsonl`, one prediction line per input line.

---

## 2. How to pick an issue

1. Open the issue tracker and filter for **`good first issue`** and **`track-A`**.
2. Pick something that matches your comfort level and is **unassigned**.
3. **Comment to claim it** before starting, so two people don't build the same thing.
4. If nothing fits, propose one: open an issue describing the gap. Small and well-scoped beats big
   and vague.

Good Track-A starter areas: adding evaluator sub-metrics, extending the leakage/forbidden-key
checks, writing schema validators, adding test cases, improving docs, or turning a baseline stub
into a real module. Later tracks (RAG, paper-level, fine-tuning) open **after** the baseline gate —
don't start there in Week 1.

**Scope rule:** one issue = one owner, one artifact, one reviewable change. If your issue is growing
tentacles, split it.

---

## 3. How to make a PR

We work through pull requests with CI and TA review. Never push to `main`.

```bash
git checkout -b your-name/short-topic
# ... make the change ...
uv run pytest -q          # must pass locally before you push
git add -A
git commit -m "Track A: <what you did> (closes #<issue>)"
git push -u origin your-name/short-topic
# then open a PR against main
```

Your PR must:

- **be small and focused** — one issue, easy to review;
- **pass CI** — tests, the leakage scan, and the baseline/template smoke runs must be green;
- **link its issue** (`closes #NN`);
- **touch no private data** — never add gold labels, stress fields, hidden inputs, or provenance
  (the `.gitignore` guards this; do not work around it);
- **include a filled** [weekly update](weekly_update_template.md) in the PR description, including
  what an LLM helped with and what you verified by hand (see the
  [LLM-use policy](llm_use_policy.md)).

**Sign-off rule:** a change enters the central repo only after **one TA signs schema/technical
validity** and **another signs scientific validity**. Expect review comments — that's the process
working, not a rejection.

---

## 4. How to report progress

Every week, post a short update (in your PR, the weekly thread, or your team channel) using the
[weekly update template](weekly_update_template.md):

- **Done** — what you actually finished.
- **PR / artifact** — link(s).
- **Blocker** — what's stopping you (name it early; a blocker raised on day 2 is cheap).
- **Next step** — the one thing you'll do next.
- **Help needed** — who/what would unblock you.

Two deliverables ride along with real systems from the start:

- a [system card](system_card_template.md) describing what your system is, and
- a line in the [contribution ledger](contribution_ledger.md) recording what you did.

---

## First-week definition of done

You've had a successful Week 1 if you can:

- [ ] run `uv run pytest`, the template, and B0 locally;
- [ ] explain the two-schema separation and why stress metadata is private;
- [ ] claim an issue and open one small PR (even a docs or test fix counts);
- [ ] post one weekly update.

That's it. You don't need to understand every metric yet — you need to be able to move safely
inside the rails. Depth comes with Tracks B–D.
