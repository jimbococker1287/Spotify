# Publication Outline

Week 11 turns the research branch into a claim-driven surface instead of a pile of benchmark artifacts.

## Working Rule

Use `make research-claims` or `python -m spotify.research_claims` to generate:

- a research-claim brief
- a publication outline
- an explicit list of missing ablations and validation gaps

## What The Claim Pack Should Do

- pick one primary claim that is strongest in the current evidence
- pick one backup claim that is still believable if the primary path slips
- connect each claim to concrete artifacts, not just intuition
- call out what is still single-run, what is repeated-seed, and what is not yet trustworthy

## Preferred Paper Shape

1. Positioning and problem statement
2. Main empirical claim
3. Backup or supporting empirical claim
4. Ablations, slice analysis, and robustness checks
5. Limitations and contract gaps

## Honesty Rule

If a benchmark lock is incomplete, abstention is effectively disabled, or a causal/friction surface looks degenerate, the claim pack should say so directly rather than promoting a weak narrative.
