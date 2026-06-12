# Creator Market Intelligence

This local-first branch rolls up saved creator report families into reusable market views instead of treating each brief as a one-off artifact.

Run:

```bash
make creator-market-intelligence
```

Or directly:

```bash
spotify-creator-market-intelligence --output-dir outputs
```

Artifacts are written under `outputs/analysis/creator_market_intelligence/`.

Main outputs:

- `scene_market_pulse.csv/json`: cross-family scene momentum, migration pull, concentration, and posture.
- `opportunity_lane_atlas.csv/json`: aggregated lanes by `scene_name x primary_driver`.
- `market_migration_network.csv/json`: strongest artist-to-artist movement routes across saved families.
- `seed_scene_bridge_atlas.csv/json`: best seed-to-scene bridge combinations.
- `release_whitespace_atlas.csv/json`: whitespace and release-timing watchlist when public release metadata is available.
- `creator_market_trend_deltas.csv/json/md`: cross-family deltas that highlight rising scenes, repeated opportunity lanes, repeated migration routes, and sparse or stale release-whitespace coverage.
- `creator_market_strategy_cards.csv/json/md`: deterministic ranked actions for repeated scenes, lanes, migration routes, and release-whitespace evidence gaps, each with proof references and a validation signal.
- `creator_market_brief.json/md`: short brief summarizing the strongest market signals and next uses.
- `creator_market_manifest.json`: artifact index plus truthful report-family counts, including how many discovered families are still partial.

Evidence pass:

```bash
make creator-evidence-lab
```

Or directly:

```bash
spotify-creator-evidence-lab --output-dir outputs
```

This writes conservative per-opportunity passports under `outputs/analysis/creator_evidence_lab/`. A signal is only `publishable` when support, recurrence, stability, metadata coverage, freshness, and claim-language gates all pass. Missing evidence becomes `watch_only` or `suppress`; raw opportunity scores remain unchanged.

Creator-label intelligence briefs now ingest the evidence manifest and passports when present. Evidence-qualified rows keep their public priority band; missing, stale, watch-only, or suppressed rows retain their raw score but are described as directional watch signals. Legacy briefs remain unchanged when the evidence artifacts are absent.

Suggested use:

1. Generate or backfill a few creator report families with `make public-insights`.
2. Run this branch to convert them into reusable market views.
3. Review `creator_market_trend_deltas.md` when multiple report families exist to separate repeated market patterns from one-off family-specific signals.
4. Feed the strongest scenes, bridges, and migration routes into creator strategy, cultural analysis, or outward packages.
5. Treat `creator_market_brief.md`, `scene_market_pulse.csv`, `opportunity_lane_atlas.csv`, and `creator_market_trend_deltas.csv` as the default downstream handoff set for branch-portfolio and outward-package surfaces when you want the creator branch to read as a repeatable market story instead of a single report family.
6. Use `creator_market_strategy_cards.md` as the bounded execution queue, and require each card's validation signal before scaling the play.
7. Run `make scope-expansion-lab` to compare creator-market readiness against the analytics, DS/quant, and research branches and surface the next creator-market implementation card in the shared queue.
