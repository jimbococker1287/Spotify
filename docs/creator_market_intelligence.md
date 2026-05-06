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
- `creator_market_brief.json/md`: short brief summarizing the strongest market signals and next uses.
- `creator_market_manifest.json`: artifact index plus truthful report-family counts, including how many discovered families are still partial.

Suggested use:

1. Generate or backfill a few creator report families with `make public-insights`.
2. Run this branch to convert them into reusable market views.
3. Feed the strongest scenes, bridges, and migration routes into creator strategy, cultural analysis, or outward packages.
4. Treat `creator_market_brief.md`, `scene_market_pulse.csv`, and `opportunity_lane_atlas.csv` as the default downstream handoff set for branch-portfolio and outward-package surfaces when you want the creator branch to read as a repeatable market story instead of a single report family.
