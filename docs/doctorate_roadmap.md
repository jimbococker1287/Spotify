# Doctorate-Level Roadmap

## Thesis Direction

Turn this repository from an experiment lab into a research platform for robust, session-based music recommendation under context shift, technical friction, and deployment risk.

The project should evolve around this central question:

How do we build session recommenders that separate user preference from playback friction, adapt under temporal drift, and know when not to make a confident prediction?

## Core Contributions

The doctorate-level version of this project should aim to deliver five contributions, not just a longer feature list:

1. A unified benchmark for session-based next-artist prediction with technical-context augmentation.
2. Retrieval-plus-reranking recommenders that scale beyond direct softmax next-item prediction.
3. Causal and counterfactual analyses that distinguish preference from friction-induced behavior.
4. Uncertainty-aware evaluation and serving with calibrated abstention.
5. Longitudinal robustness evaluation under drift, regime change, and continual retraining.

## Completion Snapshot

As of the current repository state, the roadmap requirements below are implemented in code and covered by focused tests.

- Phase 0 is complete: roadmap, experiment registry, uncertainty-aware serving, drift diagnostics, and standardized artifacts are present.
- Phase 1 is complete: deep temporal backtests, repeated-seed / significance artifacts, and robustness slices are present.
- Phase 2 is complete: retrieval, ANN validation, and reranking are present.
- Phase 3 is complete: self-supervised pretraining objectives and transfer into the retrieval stack are present.
- Phase 4 is complete: friction-aware proxy / counterfactual analyses and policy-style simulation artifacts are present.
- Phase 5 is complete: benchmark protocol, experiment registry, ablations, significance outputs, and reproducibility artifacts are present.

Immediate-build-order status:

1. Calibrated uncertainty artifacts and abstaining predictions: complete.
2. Drift diagnostics over train, validation, and test regimes: complete.
3. Deep rolling backtests with lightweight profiles: complete.
4. Dual-encoder retrieval baseline plus ANN candidate generation: complete.
5. Contextual reranker comparison against the current stack: complete.
6. Self-supervised pretraining experiments: complete.
7. Friction-aware causal analysis package: complete.

Primary implementation anchors:

- `spotify/uncertainty.py`, `spotify/evaluation.py`, `spotify/predict_service.py`
- `spotify/drift.py`, `spotify/backtesting.py`, `spotify/robustness.py`
- `spotify/retrieval.py`
- `spotify/friction.py`, `spotify/policy_eval.py`, `spotify/causal_friction.py`, `spotify/digital_twin.py`
- `spotify/research_artifacts.py`, `spotify/analytics_db.py`

Primary verification anchors:

- `tests/test_uncertainty.py`
- `tests/test_drift_and_backtesting.py`
- `tests/test_retrieval_and_friction.py`
- `tests/test_research_platform.py`
- `tests/test_governance_gate.py`
- `tests/test_predict_service_validation.py`

## Research Tracks

### Track A: Representation Learning

- Add self-supervised sequence pretraining.
- Compare masked-artist prediction, contrastive session objectives, and next-session embedding transfer.
- Reuse pretrained encoders for next-artist, skip, and session-end tasks.

### Track B: Retrieval and Reranking

- Add a two-stage recommender stack.
- Stage 1: candidate retrieval with dual encoders or ANN retrieval.
- Stage 2: contextual reranking with richer temporal and technical features.
- Compare retrieval recall, reranker gain, latency, and calibration.

### Track C: Causality and Friction Modeling

- Model how stutters, playback errors, network state, bitrate downgrades, and device context affect skips and abandonment.
- Estimate the gap between "taste-driven skip" and "friction-driven skip."
- Add intervention-style analyses: what happens if friction covariates are removed or counterfactually improved?

### Track D: Robustness, Drift, and Continual Learning

- Add deep-model rolling backtests and prequential evaluation.
- Track performance across seasonal, hourly, device, and session regimes.
- Compare static retraining, warm-start retraining, and continual adaptation.

### Track E: Risk-Aware Recommendation

- Add conformal prediction sets, abstention, and risk-aware promotion criteria.
- Promote models on calibrated utility rather than raw top-1 alone.
- Measure failure concentration, selective risk, and out-of-distribution fragility.

### Moonshot Product Extension: Group Listening / Auto-DJ

- Expand from single-user session prediction to shared-space recommendation for household, party, car, and ambient listening contexts.
- Treat this as a multi-objective coordination problem: maximize group utility while protecting the least-satisfied listener from being consistently ignored.
- Route between policy templates and safe fallback policies when disagreement, technical friction, or session-end risk spikes.
- Evaluate with group-specific metrics such as minimum member satisfaction, fairness over turns, safe-route rate, and session continuity.

## Execution Phases

### Phase 0: Research Foundation

- Add a research roadmap and experiment registry.
- Add uncertainty-aware diagnostics and serving.
- Add drift reports and richer beyond-accuracy metrics.
- Standardize outputs so future experiments produce comparable artifacts.

### Phase 1: Stronger Evaluation

- Extend temporal backtesting to deep models.
- Add repeated-seed studies and significance tests for every major comparison.
- Add temporal robustness slices: weekday/weekend, hour bucket, repeat-heavy sessions, technical-friction segments.

### Phase 2: Retrieval Stack

- Add a candidate generation layer.
- Build ANN indexes for artist embeddings.
- Add rerankers that combine sequence state, context features, and retrieval scores.

### Phase 3: Representation Learning

- Pretrain sequence encoders on self-supervised objectives.
- Fine-tune on downstream tasks.
- Compare transfer efficiency under reduced-label regimes.

### Phase 4: Causal and Counterfactual Modeling

- Add exposure-aware or semi-synthetic policy evaluation.
- Estimate friction-adjusted preference signals.
- Measure intervention outcomes on skip risk and next-artist quality.

### Phase 5: Dissertation Artifact

- Produce publication-grade tables, ablations, and figures.
- Freeze a canonical benchmark split and reproducibility protocol.
- Package the system as a reproducible research asset with benchmark scripts and paper-ready outputs.

## Concrete Build Map

### Current System Strengths

- End-to-end training, tuning, backtesting, governance, serving, and analytics already exist.
- The repository already has a strong supervised next-step prediction core.
- Technical log augmentation creates a distinctive research angle that many recommender projects do not have.

### Highest-Leverage Code Expansions

- `spotify/evaluation.py`
  Add uncertainty, drift, and richer diagnostics.
- `spotify/predict_service.py`
  Add abstention and prediction-set aware serving.
- `spotify/backtesting.py`
  Generalize temporal evaluation to deep models and continual retraining modes.
- `spotify/modeling.py`
  Add retrieval encoders, rerankers, and true graph models.
- `spotify/data.py`
  Add exposure/friction labels, temporal regime metadata, and simulator-ready exports.
- `spotify/analytics_db.py`
  Add benchmark views for uncertainty, drift, and robustness slices.
- `spotify/group_auto_dj.py`
  Add shared-space cohort planning, fairness-aware aggregation, and safe policy routing for Auto-DJ scenarios.

## Experiment Matrix

Each major paper-quality experiment should vary one axis at a time:

- Model family: classical, deep single-stage, retrieval-reranker, graph, self-supervised transfer.
- Evaluation regime: static holdout, rolling backtest, repeated seeds, drift slices.
- Objective: top-k quality, calibration, selective risk, long-tail exposure, latency.
- Context regime: normal playback, high-friction playback, device changes, session restarts.
- Adaptation mode: frozen model, warm-start retrain, continual learner.

## Minimum Publishable Claims

This project becomes dissertation-grade when it can support claims like:

- Technical-context signals improve robustness under playback-friction regimes.
- Self-supervised session representations improve sample efficiency and temporal generalization.
- Risk-aware abstention reduces high-confidence failure without collapsing useful coverage.
- Retrieval-plus-reranking outperforms direct softmax baselines under realistic candidate scale.

## Immediate Build Order

1. Add calibrated uncertainty artifacts and abstaining predictions.
2. Add drift diagnostics over train, validation, and test regimes.
3. Add deep rolling backtests with lightweight profiles.
4. Add a dual-encoder retrieval baseline plus ANN candidate generation.
5. Add a contextual reranker and compare against current best ensemble.
6. Add self-supervised pretraining experiments.
7. Add a friction-aware causal analysis package.

## Success Criteria

The project should no longer be judged by "how many models it has." It should be judged by whether it can support rigorous, reproducible answers to hard questions about recommendation quality, robustness, causality, and deployment risk.
