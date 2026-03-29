# Control Room Operating Rhythm

Weeks 5 and 6 turn the repo into an operating system, not just a run generator.

## Default Review Surface

- Open `outputs/analytics/control_room.md` first after any meaningful run.
- Treat the control room as the source of truth for promotion status, drift, robustness, friction, and the next operator move.
- If the control room says the ops-selected run is not the latest observed run, read the run-selection note before trusting any regression call.

## Standard Cadence

- Daily lane: `make schedule-run MODE=fast`
- Weekly lane: `make schedule-run MODE=full`
- Use the fast lane to confirm fixes, guard thresholds, and promotion-path stability before spending the full weekly budget.
- Only run the full lane after the control room is no longer blocked on missing instrumentation or high-priority review actions.

## Async Handoff Pack

- `outputs/analytics/control_room.md`
- `outputs/analytics/control_room_weekly_summary.md`
- `outputs/analytics/control_room_triage.md` when the control room is blocked or under attention

The control room now includes:

- explicit fast and full cadence status
- a recommended next scheduled command
- an async handoff section with the headline, share pack, and next step

## Triage Flow

- Promotion failure: compare the latest run to the last strong baseline before retraining.
- Drift warning: inspect `analysis/data_drift_summary.json` before interpreting regressions as model issues.
- Robustness or stress failure: use the triage artifact to isolate the failing slice or scenario, then re-run the fast lane.
- Instrumentation gap: backfill analysis artifacts before trusting guard thresholds or async review.

## Operator Rule

- If any high-priority control-room action is open, fix that before another full run.
- If only medium and low priorities remain, capture the decision asynchronously and keep the cadence moving.
