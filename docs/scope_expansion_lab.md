# Scope Expansion Lab

This branch-level lab keeps the four local-first expansion lanes measurable:

- `Data Engineering + Analytics Engineering`
- `Data Science + Quant`
- `Creator / Market Intelligence`
- `Research Platform`

Run:

```bash
make scope-expansion-lab
```

Or directly:

```bash
spotify-scope-expansion-lab --output-dir outputs
```

Artifacts are written under `outputs/analysis/scope_expansion/`.

Main outputs:

- `branch_expansion_scorecard.csv/json/md`: one row per expansion branch with readiness, evidence, freshness, risk, proof artifacts, top signal, top gap, and next step.
- `branch_expansion_implementation_queue.csv/json/md`: ranked development queue across the four branches.
- `branch_strategy_cards.csv/json/md`: actionable sprint cards with objective, validation command, decision rule, handoff summary, and proof artifact references.
- `strategy_cards/*.md`: one standalone markdown card per branch.
- `branch_development_cockpit.json/md`: compact operator view with branch modes, riskiest lane, and a recommended command sequence.
- `scope_expansion_manifest.json`: artifact index and branch/queue counts.

Suggested use:

1. Refresh the branch artifacts:

```bash
make analytics-db
make listener-archetypes
make quant-decision-lab
make creator-market-intelligence
make research-platform-lab
```

2. Run the scope expansion lab:

```bash
make scope-expansion-lab
```

3. Open `branch_expansion_scorecard.md` to see which branch is healthy, which one is blocked, and what proof artifacts currently support that status.

4. Open `branch_expansion_implementation_queue.md` when deciding what to build next. The queue intentionally ranks research/evidence blockers highly when they reduce downstream risk across multiple branches.

5. Open `branch_development_cockpit.md` when you want the shortest branch-by-branch operating plan.

6. Open `branch_strategy_cards.md` or an individual file under `strategy_cards/` when you are ready to work one branch in a focused pass.

7. Refresh DuckDB after the lab when you want the four-branch scorecard queryable:

```bash
make analytics-db
```

That promotes the scorecard and strategy-card fields into `scope_expansion_branch_health`, `mart_scope_expansion_health`, and the DuckDB view `scope_expansion_priority_queue`.
