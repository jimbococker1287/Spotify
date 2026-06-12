# Listening Comparison Roadmap

## Product Goal

Turn the public-listening comparison from a single report into a longitudinal
analytics product that can answer:

- How close was my listening to U.S. and global public listening on any day?
- Is that alignment rising, falling, unusually high, or unusually low?
- Which artists, tracks, podcasts, genres, and scenes explain the difference?
- How does the answer change when a new public reference year is added?
- Can dashboards and applications query the results without rebuilding them?

## Expansion Slices

### 1. Reference Catalog

Support versioned bundled references and validated user-supplied references.
This unlocks year-over-year comparisons and later Spotify Charts imports
without embedding every reference directly in the comparison engine.

### 2. Trend Intelligence

Add rolling 7-day and 30-day alignment, momentum, volatility, closer-scope
streaks, and robust anomaly flags. Daily scores then become useful signals
rather than isolated observations.

### 3. Genre And Scene Comparison

Move genre analysis into a provider-agnostic engine with explicit tag coverage
and confidence. Community tags remain a proxy and must not be described as
Spotify market share.

### 4. Explainable Narratives

Generate deterministic daily and weekly explanations from stored metrics:
strongest alignment, most distinctive dimension, concentration on public-top
entities, and projection caveats.

### 5. Dashboard Query Surface

Provide safe read-only queries over DuckDB or Parquet for date ranges,
dimensions, alignment classes, trend summaries, and notable days. This becomes
the stable boundary for a later FastAPI or visual dashboard.

## Data Contract

The existing daily comparison remains the source fact:

- `public_listening_daily_comparison`
- grain: listening date, reference edition, scope, and dimension

The existing side-by-side mart remains the core product table:

- `mart_public_listening_daily_similarity`
- grain: listening date, reference edition, and dimension

New trend and narrative outputs should derive from those tables rather than
re-reading raw streaming history. Reference and genre metadata should expose
coverage and provenance fields so projected or incomplete comparisons remain
obvious to downstream users.

## Delivery Order

1. Stabilize reference validation and trend calculations.
2. Add trend and narrative warehouse assets.
3. Expose the read-only query service.
4. Add cached genre enrichment when tag data is available.
5. Add new dated public references as official or user-supplied sources become
   available.

## Guardrails

- Never imply public duration distributions exist when only rankings exist.
- Mark historical and post-window comparisons as projections.
- Keep public reference provenance and dates queryable.
- Treat genre tags as proxy metadata with measurable coverage.
- Keep comparison features out of training data unless a separate policy and
  leakage review explicitly approves them.
