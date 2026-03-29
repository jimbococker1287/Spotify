# Benchmark Contract

Week 10 freezes the comparison protocol so future experiments can be evaluated without renegotiating the benchmark each time.

## Canonical Contract

- Contract version: `2026-week10-v1`
- Canonical benchmark profile: `small`
- Comparison mode: `repeated_seed_lock`
- Minimum repeated runs: `3`
- Significance rule: paired by `run_id` on `val_top1` with 95% confidence and `z >= 1.96`

## Locked Artifact Set

Every benchmark-eligible run should preserve:

- `run_manifest.json`
- `run_results.json`
- `benchmark_protocol.json`
- `benchmark_protocol.md`
- `experiment_registry.json`

Every aggregated benchmark-lock should preserve:

- `benchmark_lock_<id>_rows.csv`
- `benchmark_lock_<id>_summary.csv`
- `benchmark_lock_<id>_summary.json`
- `benchmark_lock_<id>_ci95.png`
- `benchmark_lock_<id>_manifest.json`
- `benchmark_lock_<id>_manifest.md`

## What Must Stay Stable

- Keep the repeated-seed benchmark profile fixed at `small` until the contract version changes.
- Do not change the run-name prefix, metric columns, or minimum seed-count rule for in-contract comparisons.
- Treat changes to sequence length, max artists, random seed policy, or data fingerprint as a new benchmark version.
- Require the full artifact pack before calling a result publication-grade.

## Working Rule

Use `make benchmark-lock` when you want a benchmark-grade comparison and treat its manifest as the source of truth for whether the comparison is actually ready.
