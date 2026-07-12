# What this is / what this is not

**SciAudit-Stress** is a course-wide **benchmark-and-systems project** for evaluating how open
LLM research assistants behave when they audit empirical ML claims **under evidence-sufficiency
stress**. Staff build the rails (schemas, evaluator, baselines, hidden splits); students build and
stress the systems and the data.

If you read only one thing, read this: the benchmark is **paper-centered and long-lived**.
**Sprint 1 (Track A) is the first *runnable slice*, not the final identity of the benchmark.**

---

## What this **is**

- **Evidence-grounded scientific auditing.** A system is given a normalized claim and a *finite,
  frozen evidence pack*, and must decide whether the evidence actually warrants the claim.
- **A traceable, structured task.** Every decision returns a verdict **plus** the evidence IDs it
  relied on, issue tags, a confidence score, and an abstention flag — so a human can inspect *why*.
- **A study of delegation.** The real question is *which parts of auditing can be delegated to an
  LLM, and where a human, a deterministic tool, calibration, or abstention is still required.*

The verdict space is deliberately sharper than ordinary fact-checking:

| Verdict | Meaning |
|---|---|
| `warranted` | The supplied evidence justifies the claim within its stated scope. |
| `overclaimed` | Evidence supports a **weaker** nearby claim, but not the claim as written. |
| `contradicted` | Evidence directly conflicts with the claim. |
| `insufficient` | The evidence pack doesn't contain enough to judge the claim. |

`overclaimed` is the conceptual center: **partial support is not warrant.**

## What this **is not**

- **Not free-form review generation.** We do not write prose "reviews." Outputs are strict,
  schema-valid, structured predictions with evidence IDs.
- **Not automatic final judgment of papers.** The unit of analysis is the **claim–evidence
  relationship**, not the paper, the authors, or the venue. Paper identities are abstracted.
- **Not an autonomous AI scientist, a paper-ranking tool, or a peer-review replacement.**
- **Not a prompt-engineering leaderboard.** Prompt tweaks alone are not a contribution; measurable
  gains in evidence localization, severe-error reduction, calibration, or abstention are.

The single most dangerous behavior we measure is the **severe false-warrant**: calling something
`warranted` when it is actually overclaimed, contradicted, or insufficient.

---

## Track A/B/C/D — the roadmap

The benchmark grows in **maturity tracks**. Each track keeps the same core interface (a system
takes an instance, returns a strict structured prediction) but widens what an "instance" and a
"system" may be. **Everyone starts in Track A.** Later tracks open only after the baseline gate.

| Track | Name | What it adds | Long-term capability it enables |
|---|---|---|---|
| **A** | **Evidence-bounded claim auditing** *(Sprint 1)* | Claim + frozen evidence pack → verdict, evidence IDs, issue tags, confidence, abstention. Bounded, deterministic, no external lookup. | The controlled core: the rail everything else is measured against. |
| **B** | Retrieval & tool-augmented auditing | Larger contexts, retrieval over evidence, deterministic numeric/table checkers, tool use. | **RAG, tool use.** |
| **C** | Paper-level & assistant workflows | Multi-claim auditing across a whole paper; assistant-style tasks that chain audits. | **AI reviewing, research assistants.** |
| **D** | Adaptation & AI-for-science | Fine-tuning (LoRA/QLoRA), multimodal (chart/table), broader scientific-reasoning systems. | **Fine-tuning, AI-for-science systems.** |

**Track A is the minimal controlled slice — not the benchmark's final identity.** It exists so
that, before Week 1, the whole pipeline runs end-to-end with a dummy system ("rails before
trains"). Its bounded, no-external-lookup design is a *measurement choice*, not a ceiling. As the
data pool grows paper-centered, Tracks B–D reuse the same schemas, evaluator, and leakage
controls to support RAG, tool use, AI reviewing, research assistants, fine-tuning, and
AI-for-science — without re-litigating the core contract.

> **Don't confuse three different "letters":**
> - **Tracks A–D** = roadmap maturity levels (this document).
> - **Tasks A–F** = sub-skills *within* one audit (verdict, evidence localization, issue tags,
>   abstention, stress robustness, human-vs-LLM comparison).
> - **Baselines B0–B4** = reference systems (control experiments), starting with B0 = always
>   `insufficient`.

---

## Why the discipline matters (leakage)

Because stress metadata trivially leaks labels (if a system sees `stress_type=evidence_removal` it
can just guess `insufficient`), the project uses a **hard two-schema separation**:

- **Student-facing input** = only `(instance_id, claim, evidence_pack)`.
- **Private evaluator metadata** = stress info, gold labels, provenance, splits — **staff-only,
  never in this repo.**

You will never commit gold labels, stress fields, or hidden data here. See
[LLM-use policy](llm_use_policy.md) and the repository `.gitignore`.

---

**Next:** read the [Week-1 guide](week1_guide.md) and start.
