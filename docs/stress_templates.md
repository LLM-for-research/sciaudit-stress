# Stress-test methodology

This document defines the first version of stress transformations for SciAudit-Stress benchmark construction.

A stress case tests whether a system can judge if a claim is warranted by the supplied evidence pack. The system must not rely on hidden metadata, paper reputation, or external knowledge.

## Verdict labels

- `warranted`: the evidence pack supports the claim within its stated scope.
- `overclaimed`: the evidence supports a weaker claim, but not the stronger transformed claim.
- `contradicted`: the evidence directly conflicts with the claim.
- `insufficient`: the evidence does not contain enough information to judge or justify the claim.

## Stress transformation types

### 1. Claim strengthening

Make the claim stronger than the evidence supports.

Example:

- Original claim: The method improves accuracy on Dataset A.
- Transformed claim: The method consistently improves accuracy on all tested datasets.
- Expected verdict: `overclaimed`

### 2. Scope expansion

Expand the claim from a narrow setting to a broader setting.

Example:

- Original claim: The method works better on small datasets.
- Transformed claim: The method works better across all dataset sizes.
- Expected verdict: `overclaimed`

### 3. Evidence removal

Remove key supporting evidence from the evidence pack.

Example:

- Original claim: The ablation shows that component X improves performance.
- Evidence after transformation: The ablation table is removed.
- Expected verdict: `insufficient`

### 4. Distractor evidence

Add evidence that is topically related but does not support the claim.

Example:

- Claim: The model improves robustness.
- Distractor evidence: Training speed comparison.
- Expected verdict: usually `insufficient` or `overclaimed`

### 5. Numeric mismatch

Change the claim so that it conflicts with reported numbers.

Example:

- Evidence: Method X = 88.1, Baseline = 90.4.
- Claim: Method X outperforms the baseline.
- Expected verdict: `contradicted`

### 6. Table/caption mismatch

Create disagreement between a table value and a claim or caption.

Example:

- Table says accuracy decreases.
- Claim says accuracy increases.
- Expected verdict: `contradicted`

### 7. Missing baseline

Make a claim about outperforming a baseline that is not present in the evidence pack.

Example:

- Claim: The method outperforms BERT.
- Evidence: only RoBERTa and T5 are reported.
- Expected verdict: `insufficient`

### 8. Weak ablation

Make an ablation claim stronger than the ablation evidence allows.

Example:

- Evidence: one small ablation result.
- Claim: the component is essential across all tasks.
- Expected verdict: `overclaimed`

### 9. Non-comparable baseline

Use evidence where systems are compared under different settings.

Example:

- Method X uses extra data, baseline does not.
- Claim: Method X fairly outperforms the baseline.
- Expected verdict: `overclaimed` or `insufficient`

## Public/private separation

Toy stress cases may include expected verdicts for documentation and testing. However, public input files must not include gold labels, stress metadata, transformation type, private provenance, or validation notes.

Student-facing input should contain only:

- `schema_version`
- `instance_id`
- `claim`
- `evidence_pack`
