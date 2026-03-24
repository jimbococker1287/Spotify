# Taste OS Demo Contract

## Purpose

This document locks the first `Personal Taste OS` demo contract so the demo can be implemented as one coherent product surface instead of a loose combination of internal modules.

The first demo scope is intentionally narrow:

- modes: `focus`, `workout`, `commute`, `discovery`
- core input: recent listening sequence plus current context
- core outputs: next-step candidates, multi-step plan, explanation, and safe fallback route
- primary audience: product review, portfolio demo, and roadmap alignment

## Demo Entry Point

The first executable contract is:

```bash
python -m spotify.taste_os_demo --mode focus --top-k 5
```

Or, if installed as a console script:

```bash
spotify-taste-os-demo --mode discovery --top-k 5
```

## Request Contract

Required fields:

- `mode`: one of `focus`, `workout`, `commute`, `discovery`
- `top_k`: number of surfaced next-step candidates

Optional fields:

- `run_dir`: explicit run path; defaults to champion alias
- `model_name`: explicit serveable model override
- `recent_artists`: pipe-separated artist tail override
- `include_video`: whether rebuilt context should include video history
- `data_dir`: raw Spotify history directory

## Response Contract

The first demo response is a single JSON payload with these top-level sections:

- `request`: normalized request values
- `current_session`: sequence tail, sequence length, and selected model
- `mode`: mode metadata and product-language description
- `top_candidates`: re-ranked next-step options for the current mode
- `journey_plan`: a multi-step plan for the selected mode
- `why_this_next`: short explanation bullets for the first planned artist
- `risk_summary`: current end-risk, friction score, and guardrail state
- `fallback_policy`: which policy the demo would route to if risk rises
- `artifacts_used`: paths to the required moonshot and serving artifacts

## Required Artifacts

The first demo requires these existing run artifacts:

- `run_results.json`
- `feature_metadata.json`
- `context_scaler.joblib`
- one serveable model artifact resolved from the run
- `analysis/multimodal/multimodal_artist_space.joblib`
- `analysis/digital_twin/listener_digital_twin.joblib`
- `analysis/safe_policy/safe_bandit_policy.joblib`

If any of the moonshot artifacts are missing, the demo should fail clearly rather than silently degrading.

## Mode Semantics

### Focus

- optimize for continuity
- minimize surprise and repeat-heavy loops
- keep energy near a steady working-session band

### Workout

- optimize for rising energy and momentum
- allow more novelty than focus mode
- avoid dead-energy transitions

### Commute

- optimize for shorter horizons and quick recovery
- stay coherent even after disruptions
- keep fallback behavior conservative

### Discovery

- optimize for controlled novelty
- stay near learned taste boundaries
- make the explanation layer especially explicit

## Module Mapping

The first demo should compose the existing stack in this order:

1. `spotify/predict_next.py` and `spotify/serving.py`
   Load the current model and next-step probabilities.
2. `spotify/multimodal.py`
   Supply artist-space continuity, novelty, and energy context.
3. `spotify/digital_twin.py`
   Supply transition behavior and end-risk scoring.
4. `spotify/safe_policy.py`
   Supply fallback routing when risk rises.

## Required Week-1 Acceptance Checklist

The first roadmap milestone is complete when all of the following are true:

- one command can produce the demo payload
- the payload shape is stable and documented
- the four modes produce visibly different ranking or planning behavior
- the response includes a human-readable explanation layer
- the response includes a safe fallback policy summary
- missing artifact failures are clear and actionable

## Explicit Non-Goals For The First Demo

- production UI polish
- real-time feedback loops beyond a single request
- full adaptive steering after skips and repeats
- creator-intelligence integration
- group-listening integration

Those belong in later roadmap weeks. The first demo only needs to prove the contract.
