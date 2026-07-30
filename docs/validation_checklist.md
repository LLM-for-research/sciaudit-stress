# Stress-case validation checklist

Use this checklist before accepting a stress case into the benchmark.

## Required fields

- [ ] The case has an `instance_id`.
- [ ] The case has an original claim.
- [ ] The case has a transformed claim.
- [ ] The case has an evidence pack with stable evidence IDs.
- [ ] The case has an expected verdict.
- [ ] The case has issue tags.
- [ ] The case has a short rationale.

## Evidence validity

- [ ] The verdict can be judged only from the supplied evidence pack.
- [ ] No external paper knowledge is required.
- [ ] Evidence IDs used in the gold label exist in the evidence pack.
- [ ] The evidence pack is not ambiguous for the expected verdict.
- [ ] Distractor evidence does not accidentally support the claim.

## Label validity

- [ ] `warranted` means the evidence directly supports the claim.
- [ ] `overclaimed` means evidence supports a weaker claim, not the full claim.
- [ ] `contradicted` means evidence directly conflicts with the claim.
- [ ] `insufficient` means evidence is missing or incomplete.

## Leakage check

- [ ] Public input does not contain `gold`.
- [ ] Public input does not contain `stress_type`.
- [ ] Public input does not contain `expected_verdict`.
- [ ] Public input does not contain private provenance.
- [ ] Instance ID is non-semantic.
- [ ] The answer cannot be guessed from metadata.

## Human verification

- [ ] A human reviewer checked the evidence.
- [ ] A human reviewer checked the expected verdict.
- [ ] A human reviewer checked issue tags.
- [ ] Any uncertainty is written in the verification note.
