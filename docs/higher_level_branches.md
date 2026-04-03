# Higher-Level Branches

Week 12 is the consolidation point in the roadmap.

The repository should now read as four primary branches rather than a broad feature list:

- `Personal Taste OS`: the product demo branch
- `Control Room`: the operating-review branch
- `Creator / Label Intelligence`: the external strategy branch
- `Recommender Safety and Research Platform`: the infrastructure and evidence branch

## Working Rule

Use `make branch-portfolio` or `python -m spotify.branch_portfolio` to regenerate the current branch map from live artifacts.

That report should answer four questions:

- who each branch is for
- how success is measured for that branch
- which artifacts prove the branch is alive
- which ideas are still deliberately deprioritized

## Distinct Audiences

- Taste OS is for product review and portfolio demos.
- Control Room is for operators and teammates reviewing model health.
- Creator Intelligence is for creators, managers, A&R, and label strategy readers.
- Safety and Research Platform is for platform builders and publication-oriented readers.

## Overlap Rule

Some systems are shared, but the branches should stay legible:

- Taste OS and Creator Intelligence can share taste-graph and multimodal foundations without becoming the same surface.
- Control Room can summarize risks and regressions from Taste OS or research work without absorbing their storytelling role.
- Safety and Research Platform can power the other branches without becoming the front-door product narrative.

## Bridge Artifact

Use `make claim-to-demo` when you need one review surface that crosses branches.

That package is not a fifth branch. It is the bridge from:

- Taste OS as product proof
- Control Room as operating proof
- Safety and Research as evidence proof

## Deprioritize For Now

- Keep Group Auto-DJ as a moonshot extension unless it becomes a primary review surface.
- Keep one-off public insight variants nested under Creator Intelligence.
- Keep extra model-family expansion behind product, safety, or benchmark value.
