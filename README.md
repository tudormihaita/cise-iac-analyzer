# tfimpact — Terraform change-impact recommendation

Given a Terraform block you just touched, `tfimpact` ranks the other blocks that will probably
need to change and the block types that will probably need to be added. The ranking function
is learned with a self-implemented Genetic Algorithm from co-change history mined from
public Terraform repositories. See [`docs/SPEC.md`](docs/SPEC.md) for the full design.

## Setup

```bash
uv sync --extra plots
# TerraDS sqlite expected at data/external/terrads/TerraDS.sqlite (see configs/default.yaml)
```

## Data collection

```bash
uv run tfimpact select-repos          # TerraDS shortlist + blobless depth check -> data/raw/repos.csv
uv run tfimpact mine                  # per-repo tables -> data/processed/<slug>/ (pinned, resumable),
                                      # then builds data/raw/dataset.csv
uv run tfimpact mine --latest         # re-mine repos with new commits, update pins
uv run tfimpact build-dataset         # re-apply dataset thresholds without re-mining
uv run tfimpact mine --only <slug>    # a single repo from repos.csv
uv run tfimpact mine --url owner/name # ad-hoc repo
uv run tfimpact clean                 # remove leftover clones in data/tmp
uv run tfimpact export-csv            # regenerate CSV copies from parquet
```

## Training and evaluation

```bash
uv run tfimpact evaluate              # grouped 5-fold CV: GA (nested grid search) + baselines (~3 h, resumable)
uv run tfimpact evaluate --methods ga_permuted,ga_negative,fixed_weights --out checks
uv run tfimpact experiments --convergence
uv run tfimpact train                 # final model -> models/ga_weights.json
uv run tfimpact export                # result tables + figures -> results/main/
```

Main result (53 repos, MRR): GA 0.746, equal weights 0.563, ROSE-style rules 0.481, random 0.118
(`results/main/summary.md`).

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format .
```
