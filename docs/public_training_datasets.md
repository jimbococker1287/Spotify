# Public Training Datasets

This module is the governance and ingestion boundary for external training
data. It deliberately does not download anything. Acquisition, license review,
and checksum capture happen before an adapter is allowed to read a local file.

The implementation lives in `spotify/public_training_data.py`.

## Canonical contracts

Interaction adapters yield pandas DataFrame chunks with:

| Field | Meaning |
|---|---|
| `source_dataset` | Stable manifest dataset ID |
| `user_id`, `session_id`, `item_id` | Source-native opaque IDs; users/sessions may be absent for policy logs |
| `timestamp` | UTC timestamp when available |
| `event_type` | `listen`, `play_count`, `exposure`, or `impression` |
| `interaction_value` | Count or source-defined interaction weight |
| `dwell_ms` | Non-negative consumption duration |
| `reward` | Logged policy outcome, normally click or long-view |
| `propensity` | Probability assigned by the logging policy, in `(0, 1]` |
| `position` | Display rank/slot |
| `explicit_positive` | Positive-label indicator when the source semantics support it |
| `policy_id`, `split` | Logging policy and source partition |
| `context_json` | Source fields not promoted into the common schema |

Metadata adapters yield item chunks containing IDs, names, genres, duration,
local audio paths, content licenses, and `metadata_json` for remaining fields.
Keeping source-native IDs prevents accidental joins across datasets.

## Required manifest

Every adapter requires a `DatasetSourceManifest` with:

- task fit and source-required columns
- dataset version and citation
- license name, license URL, access URL, and access caveats
- one `SourceFileProvenance` record per local input
- absolute local path, byte size, SHA-256, original source URL, and acquisition time
- an explicit `training_use_approved` decision, reviewer, and review timestamp

Adapters verify file size and SHA-256 by default. Set `verify_checksum=False`
only when a trusted orchestration layer already verified the same immutable
file. This switch does not bypass schema, approval, or Spotify-content checks.

Example:

```python
from spotify.public_training_data import (
    DatasetSourceManifest,
    SourceFileProvenance,
    iter_million_song_taste_profile,
)

source = "/datasets/msd/train_triplets.txt"
file_record = SourceFileProvenance.from_local_file(
    source,
    source_url="https://millionsongdataset.com/tasteprofile/",
)
manifest = DatasetSourceManifest(
    dataset_id="msd_taste_profile_2012",
    display_name="Million Song Taste Profile",
    adapter="million_song_taste_profile",
    version="local-archive-2012",
    task_fit=("implicit collaborative filtering", "retrieval"),
    required_columns=("user_id", "song_id", "play_count"),
    license_name="Dataset-specific terms; reviewed locally",
    license_url="https://millionsongdataset.com/tasteprofile/",
    access_url="https://millionsongdataset.com/tasteprofile/",
    access_caveats=("Terms apply to this exact archive and its linked components.",),
    files=(file_record,),
    training_use_approved=True,
    reviewed_by="project-owner",
    reviewed_at="2026-06-13T16:00:00+00:00",
)

for interaction_chunk in iter_million_song_taste_profile(
    source,
    manifest=manifest,
    chunksize=100_000,
):
    train(interaction_chunk)
```

## Supported source families

### Million Song Taste Profile

Adapter: `iter_million_song_taste_profile`

Expected triplets are user ID, song ID, and play count. The common release is
headerless TSV. This source fits implicit collaborative filtering and
large-catalog retrieval, but it has no event timestamps. Terms can differ
between the Taste Profile and linked Million Song components; review the exact
archive rather than assuming one umbrella license.

Official reference: <https://millionsongdataset.com/tasteprofile/>

### LFM-style logs

Adapter: `iter_lfm_listening_logs`

The adapter accepts headered files with aliases or the common headerless order:
user, artist, album, track, timestamp. It is intended for temporal and
sequential recommendation. LFM release availability and redistribution terms
have changed over time, so the manifest must describe the exact legitimately
obtained local copy. Do not silently substitute an unofficial mirror.

Dataset paper:
<https://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf>

### Music4All / Music4All-Onion

Adapters: `iter_music4all_interactions`, `iter_music4all_metadata`

Interactions and metadata remain separate streaming inputs so large joins can
be performed by the training pipeline at the appropriate grain. These sources
fit multimodal retrieval and cold-start work. Interaction, audio, image, text,
lyrics, and feature packages may carry different conditions; approve and
record each component independently.

Official archive: <https://zenodo.org/records/6609677>

### Free Music Archive

Adapter: `iter_fma_metadata`

The adapter reads either a normalized flat CSV or FMA's native two-row
`tracks.csv` header. FMA metadata is CC BY 4.0, but audio licensing is selected
per track. Preserve `content_license`, filter incompatible tracks before model
training, and retain attribution. The adapter does not assume that all FMA
audio can be used under one license.

Official repository: <https://github.com/mdeff/fma>

### KuaiRand

Adapter: `iter_kuairand_policy_logs`

KuaiRand is useful for randomized-exposure analysis, debiasing, and sequential
policy experiments. It is not music data and should not be used to learn music
item identity embeddings. The official repository declares CC BY-SA 4.0;
attribution and ShareAlike obligations must be included in the review.

Official repository: <https://github.com/chongminggao/KuaiRand>

### Open Bandit Dataset

Adapter: `iter_open_bandit_policy_logs`

Open Bandit supplies action, position, reward, logging policy, and propensity
data for off-policy evaluation. It is a fashion-domain validation corpus, not a
personal music corpus. Keep propensities and positions intact, and do not
evaluate OPE after filtering rows in a way that changes the logging-policy
support.

Official release: <https://research.zozo.com/data.html>

## Spotify training boundary

Public-dataset ingestion rejects:

- columns named like Spotify IDs, URIs, URLs, API features, or Spotify content
- values containing Spotify URIs or `api.spotify.com`, `open.spotify.com`, or
  `accounts.spotify.com` URLs
- manifests that identify Spotify Platform/API content as a source

Renaming a field does not bypass the value scan. This module is intentionally
separate from display-only public catalog enrichment and from private export
analysis. Do not route Spotify Web API responses, catalog metadata, audio
features, artwork, previews, or other Spotify Platform content into training.

Spotify's current developer policy is the source of truth:
<https://developer.spotify.com/policy>

## Operational checklist

1. Acquire the file outside the training process.
2. Retain the original archive, license, README, and citation.
3. Record its source URL and acquisition timestamp.
4. Compute SHA-256 with `SourceFileProvenance.from_local_file`.
5. Review each component's intended use and restrictions.
6. Set approval, reviewer, and review time only after that review.
7. Run an adapter with checksum verification enabled.
8. Keep the manifest beside derived datasets and model artifacts.
9. Re-review when a source version, file hash, use case, or deployment context changes.

These manifests document engineering controls and provenance. They are not
legal advice or a substitute for reviewing the applicable dataset terms.
