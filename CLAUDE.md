# CLAUDE.md — cise-iac-analyzer

Terraform change-impact recommender (course project, CI in SE / SBSE). Given a touched Terraform block, rank other blocks that will likely change and block types likely to be added. Ranking weights are learned by a **self-implemented Genetic Algorithm** over features mined from git history. **`docs/SPEC.md` is the source of truth** — read it before implementing anything and follow its milestones in order. **Current status, results and next steps: `docs/HANDOFF.md`** (read first when resuming).

## Commands
```bash
uv sync                                   # install deps
uv run tfimpact --help                    # CLI entry point
uv run tfimpact mine --only <slug>        # mine one repo (use while developing)
uv run pytest -q                          # tests
uv run ruff check . && uv run ruff format .
```

## Layout
- `src/tfimpact/collection/` — repo selection (TerraDS), bare clone with guaranteed cleanup
- `src/tfimpact/processing/` — git walk, HCL parsing, references, snapshot, diff, filters, per-repo pipeline
- `src/tfimpact/features/` — rule miner (block / type / file), propagation rules, graph, similarity (names, TF-IDF content), candidates, feature vectors
- `src/tfimpact/methods/` — baselines, `ga/` package (data, fitness, operators, search, grid, runs, method), GP / LightGBM (stretch)
- `src/tfimpact/evaluation/` — loader, splits, metrics, stats, experiments (CV runner), convergence, figures, export
- `src/tfimpact/inference/` — `recommend` for the live demo
- `configs/default.yaml` — every tunable; no magic numbers in code
- `models/` — final trained model (`ga_weights.json`, committed; written by `tfimpact train`)
- `results/<experiment>/` — committed result tables + figures (written by `tfimpact export --name <experiment>`)
- `docs/` — SPEC (design + decisions §15), THRESHOLDS (evidence), RESULTS (all results), HANDOFF (status)
- `data/` — gitignored; never commit data, clones or full results

## Invariants (do not break)
1. **No leakage:** a query at commit t may only use repo state at parent(t), rules from commits < t, and the seed block's new body. Keep the single chronological pass in `processing/pipeline.py`; the causality test must stay green.
2. **Clean state:** clones live only in `data/tmp/` and are removed in `finally` (also on errors/interrupts). Mining is resumable via `_SUCCESS` markers.
3. **Provider-agnostic code:** never hard-code `aws_`; provider filtering happens only in repo selection config.
4. **Splits by repository** for learned parameters; features are always computed per repo.
5. **GA/GP are self-implemented** (numpy only). DEAP may be used only for a one-off sanity comparison, never in the main path.
6. **Reproducibility:** every stochastic step takes a seed from config/CLI; results files record config + seed.
7. Don't change metrics, protocols or feature definitions from `SPEC.md` without asking; propose the change first.

## Conventions
- Python 3.12, type hints everywhere, dataclasses for records (`Block`, `ChangeItem`, `Query`).
- Pure functions in `processing/` and `features/` (input → output, no hidden I/O); I/O only in `pipeline.py`, `cli.py`, `evaluation/figures.py`, `evaluation/export.py`, the run-file helpers in `methods/ga/runs.py` and `inference/recommend.py`.
- Git via `subprocess` (`processing/gitio.py`), batch blob reads with `git cat-file --batch`.
- Tabular data: pandas + parquet (pyarrow), float32 for feature columns. Write tables only via `processing/io.py::write_table` (parquet always, CSV copy when `output.csv` is on). Code reads parquet only; CSV is for humans.
- Git access only via `processing/gitio.py` (plain git); do not add PyDriller.
- Vectorize anything inside GA/GP fitness (precompute X, y, query group indices).
- Logging via `rich`/`logging`, progress via `tqdm`; no prints in library code.
- Tests use small fixtures and a synthetic git repo created in a tmp dir; no network in tests.

## Workflow
- Work milestone by milestone (SPEC §13); after each, run tests and the pipeline on **one** repo before scaling.
- When a milestone is done, summarize results/numbers and what's next, and record them in `docs/RESULTS.md`.
- If a library behaves differently than SPEC assumes (e.g. python-hcl2 output format, TerraDS column names), inspect it, adapt the code, and note it in SPEC under "Implementation notes".
