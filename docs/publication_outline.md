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
- prefer supporting-artifact paths that stay portable with the `outputs/` bundle instead of only workspace-absolute references
- call out what is still single-run, what is repeated-seed, and what is not yet trustworthy

## Preferred Paper Shape

1. Positioning and problem statement
2. Main empirical claim
3. Backup or supporting empirical claim
4. Ablations, slice analysis, and robustness checks
5. Limitations and contract gaps

## Honesty Rule

If a benchmark lock is incomplete, abstention is effectively disabled, a causal/friction surface looks degenerate, or supporting artifacts only resolve through workspace-specific paths, the claim pack should say so directly rather than promoting a weak narrative.

## Readiness Rule

- `submission_readiness` should stay conservative whenever benchmark evidence is incomplete or any blockers remain open.
- `analysis_ready` should mean the branch can survive an external reading pass without hiding known contract gaps.
- `promising_but_unlocked` is the correct state when the story is interesting but still depends on unfinished benchmarks, missing ablations, or fragile artifact references.
