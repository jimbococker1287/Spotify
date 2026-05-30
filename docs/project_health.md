# Project Health Review

`spotify.project_health` is the whole-repository evaluator for deciding what to improve next. It scans the major project surfaces, checks whether each one has code anchors, tests, docs, runnable commands, and named artifacts, then writes a scorecard and development queue under `outputs/analysis/project_health/`.

Run it with:

```bash
make project-health
```

or:

```bash
python -m spotify.project_health --project-root . --output-dir outputs
```

## Outputs

- `project_health_scorecard.csv/json`: one row per major surface with health, risk, proof paths, top gap, and next step.
- `project_development_queue.csv/json`: ranked initiatives for improving, expanding, or stabilizing the project.
- `repository_hygiene.json`: generated-file and OS-noise checks, including `build/lib` and `.DS_Store` visibility.
- `project_health_review.md`: a human-readable summary for review sessions.
- `project_health_manifest.json`: paths and top-line metrics for downstream automation.

## Review Rule

Use the scorecard before adding a new branch of work. If a proposed feature does not improve one of the scored surfaces or create a clearly missing anchor, tighten the thesis before implementing it.
