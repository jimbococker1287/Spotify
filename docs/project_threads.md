# Project Threads

## Purpose

The six threads added in the latest expansion are no longer just loose ideas. They now form a coherent portfolio with shared requirements:

- thesis: each thread needs a clear reason to exist
- implementation: each thread needs code anchors
- verification: each thread needs tests or inherited integration coverage
- surface: each thread needs either a user entry point or an explicit integration-only role
- artifacts: each thread needs named outputs that make progress legible
- docs: each thread needs top-level framing in repository docs

This document audits those requirements and expands the scope of each thread so future work stays coherent.

## Audit Snapshot

| Thread | Thesis set | Code set | Tests set | Surface set | Artifacts set | Docs set | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Personal Taste OS | Yes | Yes | N/A umbrella | Yes | Yes | Yes | Set |
| Recommender Safety Platform | Yes | Yes | Yes | Yes, integration library | Yes | Yes | Set |
| Control Room | Yes | Yes | Yes | Yes, CLI + Make target | Yes | Yes | Set |
| Creator Label Intelligence | Yes | Yes | Yes | Yes, `public_insights` subcommand | Yes | Yes | Set |
| Group Auto-DJ | Yes | Yes | Yes, via moonshot lab coverage | Yes, moonshot integration | Yes | Yes | Set |
| Doctorate-Level Roadmap | Yes | Yes, via anchor map | N/A strategy thread | Yes, strategy doc | Yes | Yes | Set |

## 1. Personal Taste OS

This is the umbrella product thread. Its job is to keep the repository readable as one system instead of a pile of disconnected experiments.

Current anchors:

- Thesis and build order in `docs/personal_taste_os.md`
- Product framing in `README.md`
- Package and CLI descriptions in `pyproject.toml` and `spotify/cli.py`
- Cross-cutting outputs in `outputs/runs/<run_id>/`, `outputs/history/`, and `outputs/analytics/`

Expanded scope:

- Focus mode should prioritize continuity, low friction, and low-surprise arcs for long work sessions.
- Workout arcs should behave like energy ramps with bounded repetition and stronger fallback routing when session-end risk rises.
- Commute mode should optimize for shorter horizons, quick recovery after skips, and stronger mid-session adaptation.
- Discovery mode should push novelty within taste-safe boundaries rather than treating novelty as an isolated ranking boost.
- "Why this next" should become the default explanation layer across prediction, journey planning, and public-insight surfaces.
- Adaptive playlist steering should become the visible control loop that ties the digital twin, friction analysis, and safe policy routing together.

What success looks like:

- A single demo flow can show recent history, current mode, next-step plan, explanation, and safe fallback path.
- New modules can be explained in one sentence as improving taste modeling, session planning, explanation, or safe delivery.

## 2. Recommender Safety Platform

This is the reusable infrastructure thread. Its job is to turn Spotify-specific quality controls into a generic recommender safety layer.

Current anchors:

- Generic utilities in `spotify/recommender_safety.py`
- Spotify integrations in `spotify/backtesting.py`, `spotify/drift.py`, `spotify/governance.py`, and `spotify/evaluation.py`
- Focused verification in `tests/test_recommender_safety_platform.py`
- Product framing in `README.md`

Expanded scope:

- Treat temporal backtesting as a first-class benchmark runner for arbitrary sequence recommenders, not only the current Spotify stack.
- Treat drift analysis as a reusable diagnostics layer over context features, targets, and custom segment extractors.
- Treat promotion gating as a policy engine that can enforce utility thresholds and risk caps together.
- Treat conformal abstention as a serving contract: models can refuse unsafe predictions instead of always emitting a top-1 guess.

What success looks like:

- Another recommender project could import this layer without depending on Spotify-specific training code.
- Safety reports become comparable across model families, retrieval stacks, and future product surfaces.

## 3. Control Room

This is the operations thread. Its job is to summarize the state of the system in product language instead of forcing someone to inspect raw CSVs and JSON files.

Current anchors:

- Report builder and CLI in `spotify/control_room.py`
- Make target `make control-room`
- Console script `spotify-control-room`
- Pipeline integration in `spotify/pipeline.py`
- Tests in `tests/test_control_room.py` and `tests/test_control_room_cli_smoke.py`
- Output artifacts in `outputs/analytics/control_room.json` and `outputs/analytics/control_room.md`

Expanded scope:

- The control room should become the default readout after important runs, not an optional afterthought.
- Portfolio health should include model quality, promotion status, drift severity, robustness gaps, friction signals, and moonshot risk signals in one place.
- "Next bets" should evolve into explicit recommendations for what to retrain, what to harden, and what to productize next.
- The control room should become the bridge between research artifacts and product decisions.

What success looks like:

- A teammate can open one file and understand what changed, what is risky, what is improving, and what to do next.
- The report is good enough to support recurring reviews or scheduled monitoring.

## 4. Creator Label Intelligence

This is the creator-facing intelligence thread. Its job is to turn private listening behavior plus public catalog metadata into a graph of scenes, migrations, adjacency, and opportunity whitespace.

Current anchors:

- Core logic in `spotify/creator_label_intelligence.py`
- Public surface in `spotify/public_insights.py` via `creator-label-intelligence`
- Tests in `tests/test_creator_label_intelligence.py`
- Command documentation in `README.md`
- Output bundle under `outputs/analysis/public_spotify/creator_label_intelligence_*`

Expanded scope:

- Adjacency should answer which artists live closest together in listener behavior and multimodal space.
- Fan migration should answer where listeners tend to move next between artists or scenes.
- Scene clustering should expose local taste neighborhoods rather than only pairwise similarity.
- Release whitespace should surface where audience appetite exists but release cadence or supply looks thin.
- Label intelligence should connect related artists, release cadence, and label concentration into one opportunity map.

What success looks like:

- The artifact reads like an A&R brief rather than a raw metadata dump.
- Seed artists, local scenes, and public-catalog opportunities can be compared in one graph.

## 5. Group Auto-DJ

This is the shared-session planning thread. Its job is to move from single-user next-step recommendation toward group coordination under fairness and safety constraints.

Current anchors:

- Planning logic in `spotify/group_auto_dj.py`
- Moonshot orchestration in `spotify/moonshot_lab.py`
- Verification in `tests/test_moonshot_lab.py`
- Summary artifacts in `analysis/group_auto_dj/group_auto_dj_plans.csv` and `analysis/group_auto_dj/group_auto_dj_summary.json`
- Roadmap framing in `docs/doctorate_roadmap.md`

Expanded scope:

- Household mode should emphasize comfort, fairness over turns, and low end-risk.
- Party mode should emphasize energy, continuity, and novelty while preventing one listener from being completely ignored.
- Car mode should emphasize low friction, lower disagreement tolerance, and faster safety routing.
- Shared-space ambient mode should emphasize continuity, comfort, and recovery when disagreement or friction spikes.
- Policy routing should decide when to stay expressive and when to fall back to safer shared-session behavior.

What success looks like:

- Group plans can be evaluated with fairness, minimum satisfaction, safe-route rate, and continuity metrics together.
- The group planner becomes a credible moonshot extension instead of a disconnected demo.

## 6. Doctorate-Level Roadmap

This is the research thread. Its job is to turn the repository into a defensible, reproducible research platform rather than a large hobby project.

Current anchors:

- Thesis, phases, and claim structure in `docs/doctorate_roadmap.md`
- Research implementation anchors across `spotify/retrieval.py`, `spotify/uncertainty.py`, `spotify/drift.py`, `spotify/backtesting.py`, `spotify/friction.py`, `spotify/policy_eval.py`, and `spotify/research_artifacts.py`
- Verification anchors in the retrieval, uncertainty, drift, governance, and research-platform test suites

Expanded scope:

- Representation learning should mature into a sample-efficiency and transfer-learning story, not just an extra model family.
- Retrieval and reranking should become the scale story for realistic candidate sets and latency constraints.
- Friction and causality should become the causal story separating preference from playback-induced behavior.
- Risk-aware abstention should become the deployment story for when the model should not act confidently.
- Drift, robustness, and benchmark protocol should become the longitudinal evaluation story that supports publication claims.

What success looks like:

- Every major dissertation claim maps cleanly to code, artifacts, tests, and reproducibility outputs.
- The roadmap can guide publication-quality experiments without redefining the repository each time new work is added.

## Recommended Operating Rule

Future additions should only enter the repository if they clearly strengthen at least one of these six threads. If a new idea cannot be placed inside this map, it probably needs tighter framing before implementation.
