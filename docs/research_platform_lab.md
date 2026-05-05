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

- `run_research_registry.csv/json`: run-level view of benchmark protocol coverage, safety-platform contract coverage, portability signals, and whether the attached claim pack looks stale relative to newer run artifacts.
- `benchmark_lock_atlas.csv/json`: benchmark-lock strength, readiness, blocker text, comparator status, and manifest freshness across saved locks.
- `research_claim_registry.csv/json`: flattened claim pack with claim readiness, blocked status, evidence coverage, missing/stale supporting-artifact paths, and next gates.
- `research_platform_maturity.json/md`: current anchor-run summary plus suggested next uses, with explicit counts for blocked claims, incomplete benchmark locks, and stale evidence references.

Truthfulness notes:

- A claim can still have a strong model-side status while the lab marks it `blocked` if its missing checks remain open or its supporting artifact paths are missing or stale.
- A benchmark lock is surfaced as `incomplete` whenever `comparison_ready` is false, and the atlas carries the first comparison blocker so the reason is visible without reopening the manifest.
- Portability is only treated as `ready` when both `benchmark_protocol.json` and `safety_platform_contract.json` are present and the contract reports reusable API groups plus Spotify wrappers.
- Freshness warnings are heuristic but deliberate: if a supporting artifact is newer than the saved claim pack, or a benchmark support file is newer than the saved manifest, the lab treats that reference as stale and asks for regeneration.

Suggested use:

1. Run `make research-claims` after a fresh full run.
2. Run this branch to compare the latest research pack against historical runs and benchmark locks.
3. Use the atlas outputs to decide whether a new result is merely interesting, actually benchmark-worthy, or still blocked by incomplete comparator evidence.
