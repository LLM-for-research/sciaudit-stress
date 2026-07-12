# Responsible LLM-use policy

This project is *about* responsible research delegation, so how we use LLMs is part of the science,
not a side note. The principle is one sentence:

> **Use LLMs as assistants, never as authorities.**

---

## What LLMs **may** do

- Help you **draft** — docs, rationales, PR descriptions, first passes at text.
- Help you **code** — scaffolding, refactors, tests, debugging.
- Help you **brainstorm** — possible issue tags, counterarguments, edge cases, designs.
- **Summarize supplied evidence** and **check arithmetic** over visible evidence.

## What LLMs **may not** do

- **Be the final labeler.** An LLM must never set a gold verdict, gold evidence set, or gold issue
  tags. Those are human-adjudicated.
- **Be trusted without verification.** Every label and every evidence judgment an LLM proposes must
  be **checked by a human** against the actual evidence before it counts.
- **See hidden or private material.** Never paste hidden-evaluation inputs, gold labels, stress
  metadata, or provenance into an external chatbot or paid API. Hidden material stays in the
  approved, controlled environment.
- **Bring in outside evidence** for a bounded audit task. Track A judges *only* from the supplied
  evidence pack; asking an LLM to "look it up" breaks the measurement.

## The rule for evaluated systems

Submitted, *evaluated* systems must run on **approved open/free/local models only** — no paid or
hidden external APIs. (You may privately use any LLM as a *development* assistant, as long as you
disclose meaningful use and verify outputs.)

---

## What "verify" concretely means

Before you accept an LLM's output as fact:

| The LLM claimed… | You verify by… |
|---|---|
| a verdict / issue tag | re-reading the evidence pack yourself and confirming it follows |
| a number or comparison | checking the value against the evidence text/table |
| a citation or reference | confirming it actually exists and says what's claimed |
| working code | running it — tests pass, output is schema-valid |

Common failure modes to watch for: hallucinated citations, wrong evidence attribution,
percent-vs-percentage-point confusion, and overconfident conclusions.

---

## Disclosure

Every weekly update and every real system carries a short delegation record. In your
[weekly update](weekly_update_template.md) and [system card](system_card_template.md), state:

- what you delegated to an LLM (a category, not a full transcript),
- what you verified by hand,
- what you accepted, corrected, or rejected, and
- any failure you observed.

Honest disclosure of a rejected LLM output is a *positive* signal — it's exactly the delegation
judgment this project studies.
