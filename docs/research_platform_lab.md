# Research Platform Lab

This branch turns completed runs, benchmark locks, and claim packs into a reusable local research registry.

Run:

```bash
make research-platform-lab
```

Or directly:

```bash
spotify-research-platform-lab --output-dir outputs
```

Artifacts are written under `outputs/analysis/research_platform_lab/`.

Main outputs:

- `run_research_registry.csv/json`: run-level view of benchmark protocol coverage, safety-platform contract coverage, and research maturity.
- `benchmark_lock_atlas.csv/json`: benchmark-lock strength, readiness, leading models, and significant pair counts across saved locks.
- `research_claim_registry.csv/json`: flattened claim pack with readiness, evidence coverage, and next gates.
- `research_platform_maturity.json/md`: current anchor-run summary plus suggested next uses.

Suggested use:

1. Run `make research-claims` after a fresh full run.
2. Run this branch to compare the latest research pack against historical runs and benchmark locks.
3. Use the atlas outputs to decide whether a new result is merely interesting or actually benchmark-worthy.
