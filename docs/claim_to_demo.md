# Claim To Demo

This is the next-level review surface after Weeks 1-13.

The goal is simple: connect the strongest user-facing demo to the strongest research claim without making the reader assemble that bridge themselves.

## Working Rule

Use `make claim-to-demo` or `python -m spotify.claim_to_demo` to build the flagship package from live artifacts.

That package should do four things:

- pick the strongest Taste OS example for the current primary claim
- show whether product, ops, and research artifacts are aligned to the same run
- surface the small set of metrics that make the claim believable
- give a short talk track for product, operator, and research review

## Output

The package should write:

- `outputs/analysis/claim_to_demo/claim_to_demo.md`
- `outputs/analysis/claim_to_demo/claim_to_demo.json`
- `outputs/analysis/claim_to_demo/claim_to_demo_talk_track.md`

## Honesty Rule

If the Taste OS showcase, control room, and research claim pack come from different runs, the package should say so explicitly.

This is a coherence layer, not a marketing rewrite.
