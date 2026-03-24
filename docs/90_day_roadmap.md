# 90-Day Roadmap

## Goal

Over the next 90 days, this repository should move from a strong technical stack to a clearly legible higher-level asset with four visible outcomes:

- a real `Personal Taste OS` product demo
- a recurring operator workflow via the control room
- a polished creator / label intelligence surface
- a reusable research-and-safety platform story

This roadmap assumes we sequence work for leverage, not for maximum parallelism. The order is:

1. product proof
2. operating rhythm
3. creator intelligence expansion
4. platform / research hardening

## North Star By Day 90

By the end of this roadmap, the repo should support all of the following:

- one end-to-end Taste OS demo flow with mode-aware planning and explanation
- one recurring control-room review artifact that makes model health legible
- one polished creator-intelligence output that reads like strategy, not raw analytics
- one cleaner recommender-safety story that can be reused outside the Spotify package
- one frozen benchmark and experiment protocol that supports publication-grade work

## Phase 1: Product Proof

### Week 1: Define the demo contract

Primary outcome:

- Lock the first demo scope for `focus`, `workout`, `commute`, and `discovery`.

Milestones:

- Define a single demo input contract: recent history, optional context, optional mode.
- Define a single demo output contract: next-step candidates, multi-step plan, explanation, and fallback route.
- Decide which artifacts are required for the demo to be considered complete.
- Write a short acceptance checklist for the demo flow.

Exit criteria:

- We can describe the end-to-end demo in one paragraph.
- We know exactly which modules power each stage.

Primary anchors:

- `docs/personal_taste_os.md`
- `spotify/journey_planner.py`
- `spotify/digital_twin.py`
- `spotify/predict_service.py`

### Week 2: Build the first unified demo path

Primary outcome:

- Stitch together one visible Taste OS path instead of separate technical modules.

Milestones:

- Build a thin orchestrator around prediction, planning, explanation, and safe fallback.
- Make mode selection explicit in the interface or CLI contract.
- Ensure the output format is readable by a non-ML audience.
- Capture at least two representative session examples.

Exit criteria:

- One command or script can run the full Taste OS flow.
- The output clearly differs by mode.

### Week 3: Add adaptive steering

Primary outcome:

- The product story shifts from static recommendation to active session steering.

Milestones:

- Add a re-plan path after skip, repeat, or friction spike.
- Expose “why this changed” in user language.
- Show how safe-policy routing changes behavior under risk.
- Add a lightweight artifact or transcript for adaptive-session examples.

Exit criteria:

- The demo can recover from at least one session disruption.
- Re-planning is visible, not hidden in metrics only.

### Week 4: Polish the product narrative

Primary outcome:

- The repo can now present a coherent product story.

Milestones:

- Tighten the README and demo narrative around Taste OS, not generic training.
- Add one short walkthrough doc or notebook for the demo.
- Select 3-4 canonical screenshots, markdown summaries, or transcripts for sharing.
- Define what not to add unless it strengthens the Taste OS story.

Exit criteria:

- A new reader can understand the product thesis in under five minutes.

## Phase 2: Operating Rhythm

### Week 5: Make the control room the default review surface

Primary outcome:

- `control room` becomes the first artifact to inspect after significant runs.

Milestones:

- Expand the operating checklist around `outputs/analytics/control_room.md`.
- Ensure the control room highlights quality, drift, robustness, friction, and promotion status.
- Add a simple “what changed since last strong run” section if missing.
- Define a weekly review ritual using the control-room output.

Exit criteria:

- One artifact can answer what improved, what regressed, and what to do next.

Primary anchors:

- `spotify/control_room.py`
- `spotify/pipeline.py`

### Week 6: Add recurring review semantics

Primary outcome:

- The repo starts behaving like an operating system, not just a run generator.

Milestones:

- Standardize a recurring run cadence such as nightly fast or weekly full.
- Add clearer “next bets” language tied to product, safety, and research actions.
- Document a triage flow for drift, failed promotion, or friction spikes.
- Make sure the report can support asynchronous review.

Exit criteria:

- Another teammate could review the system state without replaying raw outputs.

## Phase 3: Creator / Label Intelligence

### Week 7: Turn creator intelligence into a finished artifact

Primary outcome:

- `creator-label-intelligence` reads like a strategy brief.

Milestones:

- Tighten output sections around adjacency, migration, scenes, whitespace, and opportunities.
- Improve the markdown storytelling order.
- Standardize a short executive summary at the top of each output.
- Collect 2-3 example runs with different seed-artist styles.

Exit criteria:

- The output is understandable to a creator, manager, or label analyst.

Primary anchors:

- `spotify/creator_label_intelligence.py`
- `spotify/public_insights.py`

### Week 8: Expand the creator surface

Primary outcome:

- The creator work becomes a real branch of the repo, not a one-off command.

Milestones:

- Add comparison views across scenes or seed groups.
- Add a clearer opportunity ranking rubric.
- Connect release cadence, label concentration, and listener migration more explicitly.
- Decide whether to package this as a standalone report family or keep it nested inside public insights.

Exit criteria:

- We can explain the creator-intelligence product in its own right.

## Phase 4: Platform and Research Hardening

### Week 9: Cleanly frame the recommender-safety layer

Primary outcome:

- The safety layer is legible as reusable infrastructure.

Milestones:

- Separate Spotify-specific framing from generic safety primitives in docs.
- Define the minimum public API for temporal backtest, drift, promotion gate, and abstention summaries.
- Add a short “how to reuse this outside Spotify” document or examples.
- Identify any remaining Spotify-coupled assumptions in the safety layer.

Exit criteria:

- A reader can see the platform story without reading the full training pipeline.

Primary anchors:

- `spotify/recommender_safety.py`
- `docs/project_threads.md`

### Week 10: Freeze the benchmark contract

Primary outcome:

- The project can support serious comparison work.

Milestones:

- Freeze a canonical benchmark configuration and protocol.
- Lock the default artifact set for benchmark runs.
- Tighten benchmark-lock and significance reporting expectations.
- Document what must remain stable for publication-grade comparisons.

Exit criteria:

- Future experiments can be compared without redefining the protocol each time.

Primary anchors:

- `docs/doctorate_roadmap.md`
- `spotify/research_artifacts.py`
- `spotify/backtesting.py`

### Week 11: Strengthen the research claims

Primary outcome:

- The repo starts looking like a paper engine instead of an ambitious engineering project.

Milestones:

- Pick one primary paper claim and one backup claim.
- Align the evaluation tables and artifacts needed to support those claims.
- Identify any missing ablations, slice analyses, or robustness checks.
- Create a short publication-outline doc if helpful.

Exit criteria:

- There is a believable path from current artifacts to a submission-ready narrative.

### Week 12: Consolidate the higher-level branches

Primary outcome:

- Product, platform, creator, and research branches now feel related rather than scattered.

Milestones:

- Reconcile overlaps between Taste OS, creator intelligence, and the safety platform.
- Make sure each branch has a distinct audience and success metric.
- Prune or deprioritize features that do not strengthen one of the main branches.
- Update top-level docs so the repo reads consistently.

Exit criteria:

- The repo has a clean hierarchy of bets instead of a broad feature list.

## Phase 5: Day-90 Integration

### Week 13: Prepare the outward-facing package

Primary outcome:

- The repo is ready to be shown, pitched, or used as a foundation for a larger initiative.

Milestones:

- Finalize the Taste OS demo package.
- Finalize one control-room review sample.
- Finalize one creator-intelligence sample artifact.
- Finalize one benchmark / safety showcase artifact.
- Write a concise top-level summary of the repo’s four big branches.

Exit criteria:

- You can show this repo as a product demo, an intelligence tool, a safety platform, and a research system.

## Delivery Checklist

By Day 90, we should have:

- one canonical Taste OS demo flow
- one recurring control-room review workflow
- one creator-intelligence artifact family with polished markdown output
- one clearly documented reusable safety layer
- one frozen benchmark protocol and a shortlist of publishable claims

## Suggested Weekly Cadence

- Monday: pick the week’s one non-negotiable outcome
- Wednesday: sanity-check whether the work is still strengthening a core branch
- Friday: produce one artifact, not just code

## Priority Rule

If a new idea appears during these 90 days, it should only be added if it clearly strengthens one of these branches:

- Taste OS product
- control-room operating layer
- creator / label intelligence
- recommender-safety / research platform

If it does not, it should wait.
