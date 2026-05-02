# Show-Ready Maintenance

Weeks 15-16 treat the Day-90 package as a living operating surface, not a one-time export.

The goal is to keep the repo honest and current after the closeout pass:

- backfill legacy artifacts that were created before the richer package format existed
- verify that product, ops, and research still point at the same review anchor
- keep the canonical package fresh after new runs

## Use

Backfill older creator and research artifacts, then refresh the package:

```bash
make show-ready-backfill
```

Audit the live package for anchor alignment, freshness, and cadence:

```bash
make show-ready-maintenance
python -m spotify.phase_readiness --scope weeks-1-16
```

Outputs:

- `outputs/analytics/show_ready_backfill/show_ready_backfill.md`
- `outputs/analytics/show_ready_maintenance/show_ready_maintenance.md`
- `outputs/analytics/weeks_1_16_readiness.md`

## Working Rule

Do not call the package current just because the launch docs exist.

The package is only truly maintained when:

- creator report families still ship a shareable reading order
- the research anchor still publishes the reusable safety-platform contract
- the Taste OS showcase, control room, and research claims line up on the same review anchor
- the canonical launch copies are refreshed after important runs

## Maintenance Loop

1. Run `make show-ready-backfill` after older artifacts become part of the launch story.
2. Run `make show-ready-maintenance` after a meaningful run or package refresh.
3. Use `weeks_1_16_readiness.md` as the single summary of whether show-readiness is still being maintained.
