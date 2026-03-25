# Taste OS Demo Walkthrough

## What This Demo Is

The `Taste OS` demo is the first unified product surface in the repository. It combines:

- serving probabilities from the current champion or chosen run
- mode-aware reranking
- a multi-step baseline plan
- risk and fallback policy summaries
- an adaptive-session transcript that shows how the plan changes after user or playback events

The main entrypoint is:

```bash
spotify-taste-os-demo --mode focus --scenario steady --top-k 5
```

Artifacts are written under:

```bash
outputs/analysis/taste_os_demo/
```

## Recommended Demo Runs

### Focus / Steady

```bash
spotify-taste-os-demo --mode focus --scenario steady
```

Use this to show the default low-surprise path.

### Discovery / Skip Recovery

```bash
spotify-taste-os-demo --mode discovery --scenario skip_recovery
```

Use this to show how the planner becomes less adventurous after the listener rejects an early suggestion.

### Commute / Friction Spike

```bash
spotify-taste-os-demo --mode commute --scenario friction_spike
```

Use this to show safe-policy routing and a more conservative replan after playback instability.

### Workout / Repeat Request

```bash
spotify-taste-os-demo --mode workout --scenario repeat_request
```

Use this to show how the planner can respect a listener’s desire for continuity without collapsing into a static loop.

## How To Read The Output

Each run produces:

- `taste_os_demo_<mode>_<scenario>.json`
- `taste_os_demo_<mode>_<scenario>.md`

The Markdown artifact is the best review surface for humans. Read it in this order:

1. `Why This Next`
2. `Top Candidates`
3. `Baseline Plan`
4. `Guardrails`
5. `Adaptive Session`

## What A Good Demo Looks Like

A good demo should make all of the following obvious:

- the selected mode actually changes the listening plan
- the planner can explain its first move in user language
- the guardrail logic is visible
- the adaptive transcript shows at least one replan after a session event

## Current Boundaries

This demo is intentionally still a thin product layer. It does not yet include:

- a front-end UI
- persistent user feedback memory
- full creator-intelligence integration
- group-listening integration
- live service endpoints beyond the current command surface

Those are later roadmap steps.
