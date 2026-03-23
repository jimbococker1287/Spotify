# Personal Taste OS

## Thesis

This repository should read as a Personal Taste OS, not as four unrelated projects sharing one package.

The product thesis:

- learn a listener's taste and session dynamics from private streaming history
- separate taste from friction and context failures
- plan better next steps instead of only ranking candidates
- explain the plan in user language
- expose policy-safe public metadata and link-outs without training on Spotify content

## Product Modes

The existing modules already point to a coherent consumer product:

- Focus mode: low-friction, low-surprise listening arcs that minimize skips and session drop-off
- Workout arcs: rising energy plans with repeat control and safe fallback routing
- Commute mode: shorter sessions with stronger continuity and quicker recovery from friction
- Discovery mode: novelty-weighted plans that still stay near learned taste boundaries
- Why this next: explanations grounded in transition likelihood, multimodal similarity, novelty, and friction risk
- Adaptive playlist steering: update the plan mid-session as the listener skips, repeats, or changes context

## Existing Building Blocks

- `spotify/digital_twin.py`: estimates transition behavior and session-end risk
- `spotify/journey_planner.py`: builds multi-step listening plans instead of one-step predictions
- `spotify/safe_policy.py`: keeps recommendations inside configurable risk limits
- `spotify/public_insights.py`: adds policy-safe explainers, release tracking, and catalog exploration
- `spotify/predict_next.py` and `spotify/predict_service.py`: expose the serving surface for product experiments

## Packaging Principle

The repo is broad because it contains the full stack:

- lab: train, tune, backtest, and benchmark
- planner: digital twin, causal friction, and policy routing
- insights: metadata, discovery, release tracking, and public catalog tooling
- serving: CLI and HTTP prediction surfaces

That breadth is acceptable if the top-level framing stays consistent: everything should either improve the taste engine, improve session planning, improve explanation, or improve safe delivery.

## Suggested Build Order

1. Keep the lab quality-green so the credibility story stays strong.
2. Package the public surfaces as installable commands and a cleaner README narrative.
3. Add a single product demo flow that shows `why this next` plus adaptive steering for one listening session.
4. Promote the best moonshot artifacts into first-class outputs instead of burying them under `analysis/`.
5. Narrow future additions to features that directly serve one of the product modes above.
