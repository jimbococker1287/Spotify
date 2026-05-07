# Creator Label Intelligence Brief

Weeks 7 and 8 turn `creator-label-intelligence` into a strategy-facing report family instead of a raw graph dump.

## What The Brief Should Answer

- Which artists are the clearest near-term opportunities?
- Which scenes are strongest right now?
- Where does listener migration already suggest momentum?
- Where does release cadence imply whitespace?

## Default Reading Order

1. Executive summary
2. Ranking view
3. Opportunity-lane view
4. Scene comparison
5. Scene strategy watch
6. Seed comparison
7. Scene vs seed comparison
8. Audience migration
9. Release whitespace
10. Supporting graph evidence

## Example Seed Styles

Use these as Week 7 showcase patterns:

- Indie / alternative: `Tame Impala|Arctic Monkeys|Phoebe Bridgers`
- Rap / crossover: `Drake|Kid Cudi|Kanye West`
- Mixed bridge set: `Tame Impala|Drake|The Strokes`

Run format:

```bash
python -m spotify.public_insights creator-label-intelligence \
  --artists "Tame Impala|Arctic Monkeys|Phoebe Bridgers" \
  --lookback-days 365 \
  --neighbor-k 5 \
  --related-limit 8
```

## Core Outputs

- `*_creator_label_intelligence.json`
- `*_creator_label_intelligence.md`
- `*_ranking_view.md`
- `*_opportunity_lane_view.md`
- `*_scene_view.md`
- `*_scene_strategy_watch.md`
- `*_seed_view.md`
- `*_scene_seed_view.md`
- `*_report_family.json`
- `*_report_family.md`
- `*_priority_shortlist.csv`
- `*_ranking_comparison.csv`
- `*_opportunity_lane_comparison.csv`
- `*_scene_comparison.csv`
- `*_seed_comparison.csv`
- `*_scene_seed_comparison.csv`
- `*_migration_watch.csv`
- `*_release_watch.csv`
- `*_scene_strategy_watch.csv`

## Week 8 Comparison Surfaces

- Ranking view: shows the ranked opportunity table with the score breakdown across adjacency, migration, release, scene, and gap components.
- Opportunity-lane view: groups the opportunity map by scene and primary driver so the brief can show where the strongest lanes are forming.
- Scene comparison: shows which scenes are strongest by local share, opportunity density, release pressure, and label concentration.
- Scene strategy watch: turns release cadence, label concentration, and migration into a short posture table for each scene.
- Seed comparison: shows which seed artists open the clearest adjacent bridge, strongest target scene, and widest scene coverage.
- Scene vs seed comparison: shows which seed/scene pairings create the cleanest opportunity lanes.

## Packaging Decision

- Keep the command nested under `spotify.public_insights`.
- Package the outputs as a standalone report family under `outputs/analysis/public_spotify/creator_label_intelligence/`.
- Normalize `*_report_family.json` around actual markdown and CSV artifacts so the opportunity-lane and scene-strategy surfaces can be re-anchored even when older families used legacy filenames.
- Re-anchor stale absolute paths by basename when the referenced files still exist inside the current report-family directory, so moved workspaces can be repaired locally.
- Record packaging freshness in the report-family manifest via `packaging_metadata`, including refresh anchors for `opportunity_lane_comparison` and `scene_strategy_watch`.

## Operator Rule

- Lead with the shortlist and executive summary when sharing externally.
- Use adjacency, scene map, and migration sections as evidence, not as the first thing a manager sees.
