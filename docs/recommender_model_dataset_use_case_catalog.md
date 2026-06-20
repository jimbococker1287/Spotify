# Recommender Model, Dataset, And Use Case Catalog

Research date: 2026-06-20

This catalog expands the Spotify Personal Taste OS project from a private next-track recommender into a broader recommender systems portfolio. The goal is not to add every fashionable model indiscriminately. The goal is to show range while keeping each model tied to a dataset, evaluation contract, explainability strategy, Optuna surface, and product use case.

## Executive Evaluation

The project is already past the "single notebook recommender" stage. It has leakage-safe track examples, temporal splits, retrieval baselines, DCN-V2 reranking, MMoE/PLE, MEANTIME-style temporal modeling, SASRec/BERT4Rec/SR-GNN modules, multimodal hooks, public-data governance, SHAP-style explainability routing, Optuna contracts, calibration, drift evidence, promotion gates, serving aliases, and safety/platform documentation.

The next improvement should be a cataloged model zoo and public benchmark lane:

1. Lock in the current promoted DCN-V2 path with a model card, deployment latency evidence, and a repeatable promotion report.
2. Add missing evidence for non-DCN models: persisted artifacts, validation/test metrics, calibration, drift, and explainability.
3. Expand the registry to cover classic, graph, autoencoder, feature-interaction, sequential, contrastive, long-sequence, multimodal, generative, and causal/policy model families.
4. Add public-dataset manifests and adapters first, then train public models only after the preflight passes.
5. Treat cross-domain datasets as benchmark demonstrations unless their license and semantics support actual transfer into the music lane.

The strongest positioning is:

> A governed recommender systems lab that can train, compare, explain, tune, monitor, and safely promote recommendation models across private listening history, public music datasets, public recommender benchmarks, multimodal content, and off-policy evaluation logs.

## Current Project Strengths

- Private track-level dataset: 147,406 leakage-safe examples, 11,092 unique tracks, 5,950 sessions.
- Retrieval and reranking: popularity, session co-occurrence, EASE, BPR-ready retrieval, DCN-V2 contextual reranking, candidate diagnostics.
- Neural sequence and multitask: MEANTIME/TiSASRec-style temporal attention, MMoE, PLE, SASRec, BERT4Rec, SR-GNN.
- Evidence and governance: Optuna, calibration, SHAP/explainability routing, drift evidence, promotion gates, public pretraining preflight, manifest governance.
- Product surface: Personal Taste OS, group Auto-DJ, creator/market intelligence, public insights, prediction CLI/API, deployment manifests.

## Current Gaps

- Retrieval models need persisted model artifacts, reproducible full-catalog evaluation, and test-set metrics.
- Neural and multitask models need the same calibration, explainability, drift, and promotion evidence DCN-V2 now has.
- Public pretraining is correctly blocked because no approved local public source manifest is ready.
- The expansion registry should separate "implemented component", "integrated trainer", "evidence complete", "serving ready", and "research-only".
- The project needs model cards and dataset cards so the breadth reads as professional scope, not a pile of experiments.
- Public comparison scripts should include non-music benchmark datasets so the repo shows general recommender fluency.

## Priority Model Catalog

### Build Now

| Family | Models | Why It Fits | Data Needed | Explainability | Optuna Surface | First Implementation Target |
| --- | --- | --- | --- | --- | --- | --- |
| Simple recommenders | Popularity, recent-popularity, Markov, item-item KNN, user-user KNN | Honest baselines and regression guards | Session-item or user-item matrix | Neighbor examples, popularity slices | k, similarity, decay | Add full test metrics and artifacts for all simple retrieval baselines |
| Linear implicit CF | EASE, SLIM, implicit ALS, BPR-MF | Strong low-cost baselines for sparse implicit feedback | Session or playlist pseudo-users, play counts | item weights, neighbors, coverage | regularization, factors, iterations, negatives | Finish EASE/ALS/BPR orchestration and model persistence |
| Two-tower retrieval | Matrix factorization, dual encoder, DSSM-style user/item towers | Candidate recall owner for reranking | Ordered histories, positives, negatives, item features | nearest neighbors, leave-one-event-out, coverage | embedding dim, sampler, temperature, loss | Add frozen candidate-source artifacts and ANN recall/latency reports |
| Feature-cross rerankers | DCN-V2, Wide & Deep, DeepFM, xDeepFM, AutoInt, DLRM | Production-style tabular ranking over sparse/context features | candidates, retrieval scores, context, labels | SHAP/ablation, cross-feature audit | cross layers, deep units, dropout, LR | Add Wide & Deep/DeepFM/AutoInt as challengers to promoted DCN-V2 |
| Sequential transformers | SASRec, BERT4Rec, TiSASRec, MEANTIME | Best fit for next-track and session continuation | ordered histories, timestamps, masks | attention diagnostics plus position ablation | sequence length, heads, layers, dropout | Re-run SASRec/BERT4Rec/MEANTIME on identical track splits |
| Multitask ranking | MMoE, PLE, shared-bottom, ESMM-style conversion heads | Skip, dwell, repeat, explicit-positive, session-end modeling | weak labels and missing-label masks | per-head attribution, loss conflict report | experts, towers, loss weights | Add calibration/drift/evidence parity for MMoE and PLE |
| Public benchmark adapters | RecBole/Microsoft-style dataset contracts | Shows general recommender competence | MovieLens, Amazon, MIND, Yelp, KuaiRand, Open Bandit | dataset cards and metric cards | framework-neutral study metadata | Add canonical manifests and smoke loaders before training |

### Build After Evidence Parity

| Family | Models | Why It Fits | Data Needed | Explainability | Optuna Surface | First Implementation Target |
| --- | --- | --- | --- | --- | --- | --- |
| Graph collaborative filtering | LightGCN, NGCF, PinSage, LightGCL | Strong user-item graph baselines, especially public multi-user data | public user-item graphs, optional item metadata | graph neighborhoods, propagation depth, exposure slices | layers, embedding dim, edge dropout | Implement LightGCN on approved public interaction data |
| Autoencoder recommenders | MultiVAE, RecVAE, DAE, CDAE | Strong top-N baselines on implicit user-item matrices | user-item matrix with enough users | reconstruction slices, latent-neighbor probes | latent dim, beta, dropout, annealing | Add RecVAE/MultiVAE on MovieLens/Amazon public lane |
| Session graph models | SR-GNN, GCE-GNN, TAGNN | Good for short session transitions and playlist continuation | session graphs, ordered item transitions | transition graph examples | hidden dim, graph steps, dropout | Compare SR-GNN with SASRec on session-only splits |
| Contrastive sequence pretraining | S3-Rec, DuoRec, CL4SRec, CoSeRec, FEARec | Useful when labels are sparse and histories are long | unlabeled sequences, augmentations, item attributes | augmentation collision report, representation uniformity | mask rate, temperature, augmentation weights | Wire existing pretraining utilities into fine-tuning deltas |
| Interest networks | DIN, DIEN, DSIN, BST | Production-flavored behavior interest extraction | candidate item, history events, context | activated history events and ablations | attention units, interest dims, sequence len | Add DIN/BST as a candidate-reranker challenger |
| Debiasing and causal ranking | IPS/SNIPS/DR, propensity-aware ERM, causal forests, uplift ranking | Validates recommendation policy claims | logged action, reward, propensity | overlap and ESS diagnostics | clipping, estimator choice, nuisance models | Connect KuaiRand/Open Bandit adapters to safe_policy |

### Research Flex

| Family | Models | Why It Fits | Data Needed | Explainability | Optuna Surface | First Implementation Target |
| --- | --- | --- | --- | --- | --- | --- |
| Long-sequence state-space | Mamba4Rec, SS4Rec, SSD4Rec, TiM4Rec | Shows current long-history modeling beyond transformers | sequences 256+ events, irregular gaps | history-length slices, ablation, throughput | state dim, blocks, sequence length | Keep research-only until long histories are materialized |
| Multimodal content retrieval | audio/text/image towers, gated fusion, CLAP-style frozen embeddings, Music4All/FMA features | Cold-start tracks and creator intelligence | licensed audio/features, text, artwork, metadata | modality ablation and missing-modality slices | projection dim, fusion, modality dropout | Train metadata/audio frozen-feature tower on Music4All/FMA |
| Semantic-ID generative retrieval | TIGER-style semantic IDs, GRID-style semantic-ID benchmarking, hierarchical codebooks | Impressive modern retrieval lane with generative decoding | stable item embeddings, codebooks, sequences | codebook utilization and collision reports | codebook size/depth, decoder dim | Build an offline semantic-ID experiment only after multimodal embeddings stabilize |
| LLM/prompt recommenders | P5-style text-to-text recommendation, natural-language query-to-music retrieval, RAG explainers | Conversational Taste OS and "ask for music" use cases | item text, user summaries, safe private context | citations, prompt traces, retrieval evidence | prompt templates, rerank depth, retrieval top-k | Start as retrieval-augmented interface, not model fine-tuning |
| Federated/private learning | FedNCF, on-device tower updates, DP-SGD experiments | Strong privacy story for personal listening data | client-style splits, private gradients | privacy budget and performance tradeoff | local epochs, clipping, noise | Research-only unless product becomes multi-user |

## Dataset And Training Set Catalog

### Private And Project-Native

| Dataset | Use | Good For | Caveats | Priority |
| --- | --- | --- | --- | --- |
| Spotify Extended Streaming History export | Private product truth | next-track, session steering, skip/dwell, repeats, personal taste twin | private only; do not publish raw IDs or personal artifacts | highest |
| Derived session-item matrix | Retrieval and collaborative baselines | EASE, ALS, BPR, item KNN, session co-occurrence | pseudo-users are sessions, not real multi-user CF | highest |
| Candidate/reranking examples | Ranking and promotion gates | DCN-V2, DeepFM, AutoInt, DLRM, MMoE/PLE | retrieval recall caps reranker quality | highest |
| Public-insights derived summaries | Demo and creator intelligence | safe public reports, trend narratives | display/evidence only unless source is licensed for training | medium |

### Public Music And Audio

| Dataset | Use | Good For | Caveats | Priority |
| --- | --- | --- | --- | --- |
| Million Playlist Dataset | Playlist continuation and large music graph benchmarking | playlist continuation, graph retrieval, session/playlist CF | AIcrowd page says direct download is no longer available; access must go through Spotify Research; contains Spotify IDs, so keep it out of training unless terms explicitly permit this project use | high if access approved |
| Million Song Dataset Taste Profile | User-song play-count CF | ALS/BPR/EASE/LightGCN benchmarks | no timestamps, metadata terms can differ by component | high |
| LFM-1b / LFM-style logs | Temporal music recommendation | sequential models, long-history evaluation | availability and redistribution terms vary by release | high if acquired legitimately |
| ListenBrainz dumps | Open listening-history research | sequence modeling, public scrobble baselines | verify current dump license and schema; entity matching to private catalog is unsafe by default | high after review |
| Music4All-Onion | Multimodal music recommendation | audio/text/image feature towers, cold-start, content fusion | component-level license/provenance required | high |
| Free Music Archive | Audio representation and genre/cold-start work | audio encoders, genre slices, cold-start retrieval | metadata is easy; per-track audio license must be preserved and filtered | high |
| MusicBrainz / AcousticBrainz-style metadata | Open metadata enrichment | entity metadata, artist/recording graph, creator intelligence | use licensing-aware metadata; avoid accidental joins to Spotify API-derived content | medium |
| Last.fm / HetRec | Legacy music recommendation benchmarks | user-artist CF, social/genre side info | old and sometimes mirror-dependent | medium |

### General Recommender Benchmarks

| Dataset | Use | Good For | Caveats | Priority |
| --- | --- | --- | --- | --- |
| MovieLens 32M / 20M / 1M | Standard recommender benchmark | MF, KNN, EASE, VAE, LightGCN, RecBole-style comparisons | movie domain; usage terms in README | high |
| MovieLens Tag Genome | Content/tag-aware recommender demos | hybrid recommenders, explainability | not music, but excellent for explainable recommendation | medium |
| Amazon Reviews 2023 | Large-scale text/item recommendation | sequential, multimodal, language-item retrieval, category transfer | huge; start with CDs/Vinyl, Digital Music, Movies/TV, Books, or Electronics | high |
| MIND | News recommendation | impression logs, clicked/non-clicked slates, text encoders | research license; news domain | medium |
| Yelp Open Dataset | Local business recommendation | review text, geo/context, rating prediction, cold-start | review authenticity and terms require care | medium |
| Criteo/Avazu/Kaggle CTR | CTR/feature interaction benchmarks | DeepFM, xDeepFM, AutoInt, DLRM | advertising domain, not personalized music | medium |
| Steam/Goodreads/Book-Crossing | Cross-domain recommenders | public user-item baselines and text metadata | terms and availability vary | low-medium |

### Policy, Causal, And Robustness

| Dataset | Use | Good For | Caveats | Priority |
| --- | --- | --- | --- | --- |
| KuaiRand | Randomized-exposure sequential recommendation | OPE, debiasing, multi-feedback, long sequences | video domain, CC BY-SA obligations | high |
| KuaiRec | Fully observed exposure evaluation | missing-not-at-random evaluation, simulator checks | video domain, small item universe | medium |
| Open Bandit Dataset | Logged bandit feedback | IPS/SNIPS/DR, contextual bandits, slate OPE | fashion domain with item actions, positions, propensities | high |
| ZOZO SHIFT15M | Distribution shift | robustness and drift demos | large download and fashion domain | low-medium |

## Use Case Expansion

| Use Case | Best Models | Data | Product Surface |
| --- | --- | --- | --- |
| Next-track prediction | SASRec, BERT4Rec, MEANTIME, DCN-V2 reranker | private streaming history | prediction CLI/API |
| Playlist continuation | EASE, BPR, LightGCN, PinSage, SR-GNN, semantic IDs | MPD if approved, session matrix | playlist builder and public benchmark |
| Focus/workout/commute session steering | DCN-V2, MMoE/PLE, safe policy routing | private sessions, skip/dwell/context | Personal Taste OS modes |
| Cold-start new music | multimodal content tower, graph + metadata hybrid, RecVAE | FMA/Music4All/public metadata | new-track discovery |
| Skip-risk and friction reduction | multitask heads, causal friction, calibrated reranker | skips, dwell, reason_end, context | "why this next" and safe policy |
| Creator/label intelligence | graph embeddings, public metadata, trend models | public insights, MusicBrainz-like metadata | creator market intelligence |
| Group Auto-DJ | multi-user aggregation, fairness/diversity reranking | multiple user profiles or simulated groups | shared session planner |
| Natural-language music requests | retrieval-augmented query tower, P5-style prompt model | item text, user taste summaries | conversational Taste OS |
| Recommender safety SDK | calibration, drift, conformal, OPE, gating | any benchmark plus private lane | reusable B2B-style platform |
| Public portfolio benchmarks | RecBole/Microsoft-style runners | MovieLens, Amazon, MIND, KuaiRand, Open Bandit | publication and demo proof |

## Recommended Development Roadmap

### Phase 1 - Evidence Parity

- Promote DCN-V2 into a formal model card with data fingerprint, feature drops, calibration temperature, drift pass, and serving constraints.
- Add model artifacts and validation/test metrics for session co-occurrence, EASE, ALS/BPR, MEANTIME, MMoE, PLE, SASRec, BERT4Rec, and SR-GNN.
- Require every trained model family to emit:
  - ranking metrics
  - coverage/diversity/long-tail metrics
  - calibration if probabilistic
  - drift evidence if feature/context based
  - explainability or retrieval-neighborhood evidence
  - Optuna study metadata or an explicit no-tuning reason

### Phase 2 - Registry Expansion

- Expand `spotify.expansion_registry` beyond the current 11 families to include:
  - `item_knn_user_knn`
  - `implicit_als_bpr`
  - `wide_deep_deepfm_xdeepfm`
  - `autoint_din_bst`
  - `dlrm`
  - `lightgcn_ngcf_pinsage`
  - `multivae_recvae`
  - `gru4rec_caser_nextitnet`
  - `cl4srec_contrastive_sequence`
  - `semantic_id_generative_retrieval`
  - `p5_prompt_recommender`
  - `bandit_ope_policy`
- Track implementation state separately from evidence state.

### Phase 3 - Public Dataset Preflight

- Create manifest templates for MovieLens, Amazon Reviews 2023, MIND, MPD, ListenBrainz, Music4All-Onion, FMA, KuaiRand, Open Bandit, and Criteo/Avazu.
- Add smoke adapters that read tiny local samples into the canonical schema without downloading data.
- Keep `training_use_approved=False` until source, license, checksum, and review fields are complete.

### Phase 4 - Model Zoo Training

- Add one model from each major family before adding multiple variants:
  - ALS or BPR for matrix factorization
  - LightGCN for graph collaborative filtering
  - RecVAE for autoencoder recommendation
  - DeepFM or AutoInt for feature interaction
  - DIN or BST for candidate-aware behavior interest
  - CL4SRec or DuoRec for contrastive sequence pretraining
  - Mamba4Rec only after long histories are available
- Compare every advanced model against EASE, ALS/BPR, SASRec, and DCN-V2.

### Phase 5 - Multimodal And Generative

- Start with frozen licensed features from Music4All-Onion and FMA.
- Train content towers for cold-start retrieval before attempting generative semantic IDs.
- Build semantic-ID generation as an offline research artifact with strict collision, invalid-ID, and codebook-utilization checks.
- Build natural-language recommendation as retrieval-augmented querying first. Avoid LLM fine-tuning until the dataset, privacy, and evaluation story are solid.

### Phase 6 - Platform And Demo

- Add a model catalog dashboard or report artifact.
- Add benchmark comparison reports for private music, public music, public general recommendation, and policy/OPE lanes.
- Add model cards and dataset cards to outward-package artifacts.
- Make the demo show:
  - "simple baseline beats hype" comparisons
  - "advanced model wins with evidence" comparisons
  - "model rejected by gates" examples
  - "policy cannot be claimed without propensity support" examples

## What Not To Do

- Do not train on Spotify Platform/API metadata, audio features, preview URLs, artwork, or catalog content unless the current Spotify policy and project terms explicitly permit the intended training use.
- Do not promote long-sequence or generative models before simple retrieval baselines are locked.
- Do not mix public users/items with private Spotify export identities.
- Do not claim causal policy lift without propensities, overlap diagnostics, and effective sample size.
- Do not use attention weights as the only explanation.
- Do not let Optuna tune on the test split or against public-policy evaluation logs.

## Source Notes

- RecBole documents a broad PyTorch recommendation framework with many model families and datasets: https://recbole.io/
- Microsoft Recommenders provides examples and best practices across data prep, modeling, evaluation, tuning, and operationalization: https://github.com/recommenders-team/recommenders
- NVIDIA Merlin Transformers4Rec targets sequential and session-based recommendation: https://nvidia-merlin.github.io/Transformers4Rec/stable/
- SHAP documents Shapley-value based model explanations: https://shap.readthedocs.io/en/latest/
- Optuna documents framework-agnostic hyperparameter optimization, pruning, and parallel search: https://optuna.org/
- GroupLens MovieLens provides stable benchmark datasets, including MovieLens 32M: https://grouplens.org/datasets/movielens/
- The Spotify Million Playlist Dataset challenge documents automatic playlist continuation and current access caveats: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
- Spotify Developer Policy is the current source for Spotify Platform/API usage and training restrictions: https://developer.spotify.com/policy
- Amazon Reviews 2023 documents large-scale review, metadata, image, graph, and timestamp fields: https://amazon-reviews-2023.github.io/
- Music4All-Onion documents multimodal music features and large Last.fm listening records: https://zenodo.org/records/6609677
- FMA documents Creative Commons licensed audio and metadata for music analysis: https://github.com/mdeff/fma
- MIND documents large-scale news recommendation impressions and clicked/non-clicked events: https://msnews.github.io/
- KuaiRand documents randomized-exposure sequential recommendation logs: https://github.com/chongminggao/KuaiRand
- Open Bandit Dataset documents logged bandit feedback with actions, positions, propensities, and clicks: https://research.zozo.com/data.html
- Key papers reviewed include DCN-V2, SASRec, BERT4Rec, EASE, LightGCN, MultiVAE/RecVAE, DeepFM, xDeepFM, AutoInt, DLRM, PinSage, Mamba4Rec, SS4Rec, P5, and semantic-ID generative recommendation work.
