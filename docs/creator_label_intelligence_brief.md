# Creator Label Intelligence Brief

Week 7 turns `creator-label-intelligence` into a strategy-facing artifact instead of a raw graph dump.

## What The Brief Should Answer

- Which artists are the clearest near-term opportunities?
- Which scenes are strongest right now?
- Where does listener migration already suggest momentum?
- Where does release cadence imply whitespace?

## Default Reading Order

1. Executive summary
2. Immediate opportunity shortlist
3. Scene comparison
4. Seed comparison
5. Audience migration
6. Release whitespace
7. Supporting graph evidence

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
- `*_priority_shortlist.csv`
- `*_scene_comparison.csv`
- `*_seed_comparison.csv`
- `*_migration_watch.csv`
- `*_release_watch.csv`

## Operator Rule

- Lead with the shortlist and executive summary when sharing externally.
- Use adjacency, scene map, and migration sections as evidence, not as the first thing a manager sees.
