# Day-90 Launch

Week 14 is the closeout layer on top of the outward-facing package.

The goal is simple: declare the canonical artifacts, score the delivery checklist, and make it obvious whether the repo is ready to show as:

- a product demo
- an operating-review workflow
- a creator-intelligence tool
- a safety and research platform

## Use

```bash
make day-90-launch
```

Outputs:

- `outputs/analysis/day_90_launch/day_90_launch.md`
- `outputs/analysis/day_90_launch/canonical_artifact_manifest.md`
- `outputs/analysis/day_90_launch/delivery_checklist.md`

After the closeout pass, Weeks 15-16 keep this package fresh with:

```bash
make show-ready-backfill
make show-ready-maintenance
python -m spotify.phase_readiness --scope weeks-1-16
```

## Working Rule

This package should not invent a cleaner story than the repo has earned.

It should:

- name the canonical artifacts for each branch
- preserve any live caveats or alignment gaps
- make the Day-90 delivery checklist visible in one place

## Review Order

1. Open the front door.
2. Open the claim-to-demo bridge.
3. Follow the canonical Taste OS, Control Room, Creator Intelligence, and Safety / Research artifacts.
4. End on the delivery checklist to confirm what is show-ready versus what still needs notes.
5. Use the show-ready maintenance report to confirm the package is still aligned after later runs.
