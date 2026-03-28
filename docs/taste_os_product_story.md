# Taste OS Product Story

## Thesis In 90 Seconds

`Taste OS` is not just "predict the next artist."

It is a product layer that turns listening history, current context, multimodal taste structure, and safety routing into an active session planner. The right way to read the repository is:

1. the model proposes a candidate field
2. the mode surface chooses the right opening lane
3. the planner extends that choice into a coherent short arc
4. the adaptive transcript shows how the system changes course after user or playback events
5. the safety layer keeps the session legible when friction or risk rises

If a new reader understands those five moves, the product story is working.

## Canonical Review Order

Use the Week 3-4 showcase pack in this order:

1. `Focus / Steady`
Show the baseline. This is the "what does a good normal session look like?" example.

2. `Discovery / Skip Recovery`
Show that the planner can start on a more exploratory lane, then explain why it becomes less adventurous after rejection.

3. `Commute / Friction Spike`
Show the guardrail story. This is the proof that the system is not only optimizing taste, but also reacting to playback risk.

4. `Workout / Repeat Request`
Show that a user signal for continuity can tighten the arc without collapsing into a degenerate loop.

## What A Reviewer Should Notice

- The opening artist should differ by mode when the candidate field is close.
- The `why this next` copy should sound like product language, not feature names.
- The adaptive transcript should make replanning visible in plain English.
- The fallback policy should be understandable from the Markdown output alone.

## What Not To Add Unless It Strengthens The Story

- Do not add UI work that hides the product mechanics instead of clarifying them.
- Do not add more model families to the demo narrative unless they improve the opening choice or the recovery behavior.
- Do not expand the showcase into creator, research, or control-room material unless it directly sharpens the Taste OS thesis.
- Do not add more canonical examples than a reviewer can absorb in one short read.

## Week 3-4 Deliverables

The shareable artifact set for this stage is:

- canonical demo Markdown + JSON outputs for the four showcase scenarios
- one Taste OS showcase pack
- one steady-mode comparison artifact

Generate them with:

```bash
spotify-taste-os-showcase --top-k 5
```

Or via Make:

```bash
make taste-os-showcase EXTRA_ARGS='--top-k 5'
```
