# Data Science And Quant Lanes

This repository now has two explicit local-first expansion lanes beyond the core product and ops surfaces:

- `listener archetypes`: behavioral clustering and taste-state summaries over time
- `quant decision lab`: model and policy frontier analysis for multi-objective recommendation decisions

These lanes are intentionally private and research-oriented. They are designed to help you think better, not just ship more code.

## Listener Archetypes

Build listener archetypes from the local analytics warehouse:

```bash
make listener-archetypes
```

This writes:

- `outputs/analysis/listener_archetypes/listener_archetype_assignments.csv`
- `outputs/analysis/listener_archetypes/listener_archetype_summary.csv`
- `outputs/analysis/listener_archetypes/listener_archetype_summary.json`
- `outputs/analysis/listener_archetypes/listener_archetype_monthly.csv`
- `outputs/analysis/listener_archetypes/listener_archetype_seasonal.csv`
- `outputs/analysis/listener_archetypes/listener_archetype_seasonal.json`
- `outputs/analysis/listener_archetypes/listener_archetype_transitions.csv`
- `outputs/analysis/listener_archetypes/taste_evolution_regime_shifts.csv`
- `outputs/analysis/listener_archetypes/taste_evolution_regime_shifts.json`
- `outputs/analysis/listener_archetypes/taste_state_brief.json`
- `outputs/analysis/listener_archetypes/taste_state_brief.md`
- `outputs/analysis/listener_archetypes/taste_evolution_brief.json`
- `outputs/analysis/listener_archetypes/taste_evolution_brief.md`

Use it to answer:

- what kinds of listening states show up repeatedly in local behavior
- which states are highest-skip or most exploratory
- how taste states transition across days
- where month-over-month regime shifts are strongest
- which archetypes are most seasonal across winter / spring / summer / fall
- which archetypes should become explicit Taste OS modes or evaluation slices

## Quant Decision Lab

Build the quant decision surface from the latest completed run:

```bash
make quant-decision-lab
```

This writes:

- `outputs/analysis/quant_decision_lab/model_decision_frontier.csv`
- `outputs/analysis/quant_decision_lab/model_decision_frontier.json`
- `outputs/analysis/quant_decision_lab/policy_decision_frontier.csv`
- `outputs/analysis/quant_decision_lab/policy_decision_frontier.json`
- `outputs/analysis/quant_decision_lab/scenario_sensitivity.csv`
- `outputs/analysis/quant_decision_lab/scenario_sensitivity.json`
- `outputs/analysis/quant_decision_lab/archetype_decision_bridge.json`
- `outputs/analysis/quant_decision_lab/archetype_decision_bridge.md`
- `outputs/analysis/quant_decision_lab/scenario_utility_simulation.csv`
- `outputs/analysis/quant_decision_lab/scenario_utility_simulation.json`
- `outputs/analysis/quant_decision_lab/scenario_utility_simulation.md`
- `outputs/analysis/quant_decision_lab/quant_decision_brief.json`
- `outputs/analysis/quant_decision_lab/quant_decision_brief.md`

Use it to answer:

- which models are efficient under quality, utility, uncertainty, and speed together
- which policies survive stress tradeoffs instead of only looking good on one metric
- which scenarios create the most decision pressure
- which model / policy / scenario combinations have the strongest transparent utility score
- whether the serving model is still efficient once risk and cost are included
- how dominant, high-skip, and exploratory listener archetypes should map onto model, policy, and scenario lanes
- where high-skip or lifecycle-drift listener contexts should add notes before promotion

## Recommended Workflow

1. Refresh the analytics warehouse:

```bash
make analytics-warehouse
```

2. Build archetypes:

```bash
make listener-archetypes
```

3. Build the quant frontier:

```bash
make quant-decision-lab
```

4. Use the artifacts together:

- archetypes define the behavioral slices worth studying
- quant frontier defines the models and policies worth trusting inside those slices
- archetype decision bridge turns those two views into concrete evaluation lanes without changing downstream consumers yet
- scenario utility simulation ranks concrete model / policy / scenario combinations and carries high-skip or high-drift caveats into the local review artifact

5. Review the current recommendation set in the control room:

```bash
make control-room
```

- when `outputs/analysis/quant_decision_lab/archetype_decision_bridge.json` exists, the control-room markdown surfaces a concise DS/quant bridge block tied to the current review anchor

6. Roll the DS/quant lane into the four-branch development queue:

```bash
make scope-expansion-lab
```

- `outputs/analysis/scope_expansion/branch_expansion_scorecard.md` shows whether the quant branch is ready, attention, blocked, or missing based on frontier, policy, scenario, and archetype-bridge artifacts.
- `outputs/analysis/scope_expansion/branch_expansion_implementation_queue.md` ranks the next quant implementation against analytics, creator-market, and research-platform work.
