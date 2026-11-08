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

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format .
```
