# SPEC — Terraform Change-Impact Recommendation

Repository: `cise-iac-analyzer` · Package: `tfimpact` · Course: Computational Intelligence in Software Engineering (Project 2)

## 1. Goal

Build a tool that, when a developer touches a Terraform block (resource, data source, module call, variable, output, local), returns a **ranked list of other blocks that will probably need to change**, plus **block types that will probably need to be added**. The ranking function is **learned from mined commit history** of public Terraform repositories.

Research framing:
- **Task (T):** given one changed block (the *seed*), rank candidate co-changes.
- **Experience (E):** commit histories of 20–30 public Terraform repos, mined by us.
- **Performance (P):** MRR, Recall@k, Precision@k on *unseen* repositories and *future* commits.
- **ML/SBSE technique:** association-rule learning (unsupervised) produces coupling signals; a **self-implemented Genetic Algorithm** learns the weights that combine all signals into a ranking score (supervised via self-mined labels). Optional: Genetic Programming (non-linear formula), LightGBM ranker.
- **Closest prior work:** Bessghaier et al., *A search-based file recommendation approach for IaC evolution*, JSS 2026 (GA-IaCRec; file level, Ansible/Puppet, fixed heuristic weights α=β=0.5). Our deltas: Terraform, **block level**, **add-type predictions** (cold start), **dependency-graph features**, **learned weights**, **cross-repository evaluation**.

## 2. Scope

**Core (must ship within ~1 week, 2–3 h/day):**
- Repo selection (TerraDS-based), mining pipeline with clean-up, block-level change extraction.
- Feature computation (history, type-level, graph, lexical/locality) — leakage-free by construction.
- Baselines + own GA weight learner.
- Evaluation: grouped 5-fold CV by repo × 10 GA seeds; per-repo temporal 90/10 check; Wilcoxon + A12 + Holm.
- Ablation by feature family, learning curve.
- CLI `recommend` for live demo.

**Stretch (only after core results exist):** GP scorer, LightGBM ranker, GA-IaCRec per-query replica, provider-neutral type categories, attribute-level targets, cutoff sensitivity (20/30/50), layer partitioning (future work section only).

**Out of scope:** running `terraform init/plan/apply`, cloud access, CI log mining.

## 3. Tech stack

- Python 3.12, **uv** for env/deps.
- Runtime deps: `python-hcl2`, `pandas`, `pyarrow`, `numpy`, `networkx`, `scipy`, `typer`, `rich`, `pyyaml`, `tqdm`, `requests` (GitHub API depth check, optional).
- Dev deps: `pytest`, `ruff`.
- Stretch deps (optional group): `deap` (GP sanity check only), `lightgbm`, `matplotlib` (plots; may be core if needed for report).
- Git access via **`subprocess` + system `git`**. **PyDriller is deliberately not used**: it only exposes changed-file contents (no full-tree snapshots), has no first-parent traversal, and is slower (per-commit GitPython + diff/metric computation we don't need). Plain `rev-list --first-parent`, `diff-tree`, `ls-tree`, `cat-file --batch` cover everything.
- No database engine needed: pandas over parquet is enough for all analysis. (DuckDB is optional, only if ad-hoc SQL over many parquet files becomes handy.)

## 4. Repository layout

```
cise-iac-analyzer/
├── pyproject.toml                # uv project; console script: tfimpact = tfimpact.cli:app
├── README.md
├── CLAUDE.md
├── docs/
│   ├── SPEC.md                   # this file (design + decisions, §15)
│   ├── THRESHOLDS.md             # every threshold / setting with its evidence
│   ├── RESULTS.md                # all results so far (committed record)
│   └── HANDOFF.md                # current status + how to resume work
├── configs/
│   └── default.yaml              # all tunables (paths, filters, GA params, seeds, k values)
├── models/
│   └── ga_weights.json           # final trained model (committed; used by `recommend`)
├── results/                      # committed copies of result tables + figures (`tfimpact export`)
│   └── main/                     # cross-validated comparison (+ ablation/, learning_curve/ later)
├── data/                         # gitignored (except data/raw/repos.csv)
│   ├── external/terrads/         # TerraDS.sqlite
│   ├── raw/repos.csv             # selected repos (pinned head_sha); dataset.csv
│   ├── tmp/                      # transient bare clones (always cleaned)
│   ├── processed/<repo_slug>/    # per-repo parquet outputs
│   ├── results/                  # full evaluation outputs, ga_runs/ (resumable per-run JSON)
│   └── logs/                     # one log per CLI run
├── src/tfimpact/
│   ├── __init__.py
│   ├── cli.py                    # Typer app (all commands)
│   ├── config.py                 # load/validate YAML into dataclasses
│   ├── constants.py              # block kinds, regexes, action symbols, column names
│   ├── collection/
│   │   ├── select_repos.py       # TerraDS SQL query + optional GitHub depth check
│   │   ├── clone.py              # bare clone context manager (guaranteed cleanup)
│   │   └── utils.py              # slugify, retry, logging helpers
│   ├── processing/
│   │   ├── gitio.py              # rev-list, diff-tree, cat-file batch, ls-tree
│   │   ├── hcl_parser.py         # .tf text -> {address: Block}
│   │   ├── references.py         # body -> set of referenced addresses
│   │   ├── snapshot.py           # incremental repo state (files, blocks, ref index)
│   │   ├── diff.py               # before/after blocks -> change items
│   │   ├── filters.py            # usable-commit rules
│   │   ├── io.py                 # write_table: parquet (+ optional csv copy)
│   │   └── pipeline.py           # per-repo single pass: walk -> queries -> features -> parquet
│   ├── features/
│   │   ├── association_rules.py  # AssociationRuleMiner (instance + type level), causal updates
│   │   ├── graph.py              # proximity, direct edge, neighbour Jaccard, degree
│   │   ├── similarity.py         # name/path token similarity, locality
│   │   ├── candidates.py         # candidate generation + pre-filter
│   │   └── vector.py             # assemble normalized feature vector per (query, candidate)
│   ├── methods/
│   │   ├── base.py               # Method: fit(train repos) / rank(repo) -> Ranking
│   │   ├── baselines.py          # arm (ROSE), fixed weights, random reference
│   │   ├── ga/                   # self-implemented GA: data, fitness, operators, search
│   │   │                         #   (GA loop + random search), grid, runs, method
│   │   ├── gp.py                 # (stretch) self-implemented GP
│   │   └── lgbm.py               # (stretch) LightGBM ranker
│   ├── evaluation/
│   │   ├── metrics.py            # RR, AP, R@k, P@k, hit@k, bootstrap CI, ceiling, file-level
│   │   ├── loader.py             # load per-repo tables (model features, warm-up dropped)
│   │   ├── splits.py             # grouped k-fold by repo (balanced), inner split for tuning
│   │   ├── stats.py              # Wilcoxon signed-rank, Vargha–Delaney A12, Holm
│   │   ├── experiments.py        # main, ablation, learning curve runners
│   │   ├── convergence.py        # convergence check (generation budget)
│   │   ├── figures.py            # plots of an evaluation run
│   │   └── export.py             # copy curated results to results/<experiment>/
│   └── inference/
│       └── recommend.py          # mine a local repo on the fly, score seeds, explain
└── tests/
    ├── fixtures/                 # tiny .tf samples + a tiny git repo built in a fixture
    ├── test_hcl_parser.py
    ├── test_references.py
    ├── test_diff.py
    ├── test_rules_causality.py
    ├── test_metrics.py
    └── test_ga.py
```

## 5. Data collection

### 5.1 Repo selection (`tfimpact select-repos`)
Source: TerraDS SQLite (`Repository`, `Module`, `Resource` tables — inspect `.schema` first, column/enum names may differ).

Default criteria (all in `configs/default.yaml`):
- permissive license (already guaranteed by TerraDS), `Archived = false`
- provider contains `aws` (core run is AWS-only; code must not hard-code `aws_`)
- ≥ 20 managed resources, ordered by stars
- shortlist top ~150, then **depth check**: bare-clone (or GitHub API) and count **usable commits** (non-merge first-parent commits with 2–30 changed blocks). Keep repos with ≥ 50 usable commits. Target 20–30 repos.

Relaxation ladder if too few: lower usable commits to ≥ 30 → drop resource minimum → any provider → add repos outside TerraDS (GitHub search sliced by creation date, or orgs like `terraform-aws-modules`, `cloudposse`; tag `repo_kind = module_library`).

Output: `data/raw/repos.csv` with `slug, full_name, clone_url, stars, license, repo_kind`.

### 5.2 Mining loop (`tfimpact mine`)
For each repo in `repos.csv`:
1. Skip if `data/processed/<slug>/_SUCCESS` exists (resumable).
2. `with bare_clone(url) as path:` → clone into `data/tmp/<slug>.git`; **always `shutil.rmtree` in `finally`**, also on Ctrl-C.
3. Run `pipeline.process_repo(path)` (Section 6).
4. Write parquet files, then `_SUCCESS`.
5. Append a line to `data/processed/summary.csv` (counts, parse rate, runtime, skip reasons).

`tfimpact clean` removes `data/tmp/` leftovers.

## 6. Processing pipeline (single chronological pass per repo)

### 6.1 History walk (`gitio.py`, `snapshot.py`)
- Linear history: `git rev-list --first-parent --reverse HEAD`.
- Cap: last `max_commits` (default 2000). Initialize the snapshot from the **full tree at the start commit** (`git ls-tree -r` + `cat-file --batch` for all `*.tf`).
- Per commit: `git diff-tree -r -M --name-status <parent> <commit> -- '*.tf'` → added/modified/deleted/renamed paths; fetch new contents with `cat-file --batch`.
- Merge commits (on first-parent): apply to snapshot, **never** use as transactions.
- Only `.tf` files (ignore `.tf.json`, `.tfvars`, `.terraform.lock.hcl`).

### 6.2 Blocks (`hcl_parser.py`)
Address scheme (module scope = directory):

| Block | Address | `type` |
|---|---|---|
| resource | `{dir}::resource.{type}.{name}` | resource type |
| data | `{dir}::data.{type}.{name}` | data type (prefix `data.`) |
| module call | `{dir}::module.{name}` | `module` |
| variable | `{dir}::var.{name}` | `variable` |
| output | `{dir}::output.{name}` | `output` |
| local | `{dir}::local.{key}` (one per key) | `local` |

`Block` fields: `address, kind, type, name, file, body_norm` (canonical JSON, sorted keys), `body_hash`, `refs` (set of addresses).
Parse failures: log `(repo, commit, file, error)`, keep the file's previous blocks, report parse rate.

### 6.3 References (`references.py`)
Regex over the canonical body string, resolved **within the same directory**:
- `data.<type>.<name>` → data block; `<type>.<name>` where type matches `^[a-z0-9]+_[a-z0-9_]+$` → resource; `module.<name>`; `var.<name>`; `local.<name>`.
- `depends_on` entries are references too.
- (Nice-to-have) local module plumbing: if `module.x` has a local `source` (`./modules/y`), add edges `module.x → modules/y::var.*` for passed arguments and resolve `module.x.<out>` → `modules/y::output.<out>`.

### 6.4 Change items (`diff.py`)
For changed files, compare before/after blocks by address:
- `+` added, `-` removed, `~` modified (`body_hash` differs). Formatting-only changes disappear by construction.
- Renames = remove + add (no rename detection in core).
- Instance item = `address`; type item = `{action}{type}` (e.g. `+aws_dynamodb_table`).

### 6.5 Usable commits (`filters.py`)
Usable if: not a merge, not the start commit, 2 ≤ #items ≤ `max_items` (default 30). Store skip reasons.
Warm-up: the first `warmup_frac` (default 0.2) of usable commits update the rule miner but produce **no queries**.

### 6.6 Single-pass order per commit t (the leakage guarantee)
```
state = snapshot at parent(t); miner holds commits < t
if usable(t) and past warm-up:
    for each item q in changes(t):            # q is the seed
        candidates = generate(q, state, miner)
        features   = compute(q, candidates, state, miner, seed_body_after=q.new_body)
        labels     = changes(t) - {q}
        emit query rows
miner.update(changes(t)); state.apply(t)
```
The **only** post-commit information used for a query is the seed block's own new body (simulates the developer having just written that block).

## 7. Candidates & features

### 7.1 Candidates (`candidates.py`)
- **Existing-block candidates:** all blocks in the parent snapshot, except the seed.
- **Add-type candidates:** pseudo-items `+type:T` for every type T with `type_conf(seed_type_item → +T) > 0`.
- **Pre-filter** to `max_candidates` (default 200), keeping the union of: rule conf > 0, graph distance ≤ `prefilter_max_dist` (4), same dir, name similarity > 0; fill with highest-degree blocks. Record **candidate recall** (fraction of true labels that survived) — this caps achievable recall and must be reported.

### 7.2 Labels
- Existing candidate `c`: `label = 1` if `c` is changed (`~` or `-`) in commit t.
- Add-type candidate `+type:T`: `label = 1` if commit t adds ≥ 1 block of type T (other than the seed).

### 7.3 Features (`vector.py`) — feature set v2, all in [0, 1]

Data sources (leakage rule, §6.6): **H** = commits before t only; **P** = snapshot at parent(t); **S** = the seed's new body (removed seed: its last body). All miners learn from commit t only after its queries were built. Decisions and evidence: §15.12.

| # | Name | Family | Definition | Uses |
|---|---|---|---|---|
| 1 | `hist_cochange_conf` | history | ROSE confidence `N(q,c)/N(q)` over earlier transactions with 1–30 items (0 if the seed has no history) | H |
| 2 | `hist_cochange_count` | history | support `log1p(N(q,c)) / log1p(max in query)` | H |
| 3 | `hist_cochange_recency` | history | Σ over co-change commits of `exp(-Δcommits/τ)`, τ = 100, divided by the query max | H |
| 4 | `hist_cand_recent_changes` | history | candidate's change count in the last 200 commits, divided by the query max | H |
| 5 | `hist_file_cochange_conf` | history | file-level rule `N(file(q), file(c)) / N(file(q))`; same file: share of the file's transactions in which ≥ 2 of its blocks changed | H |
| 6 | `hist_type_cochange_conf` | history | type rule `action+type(q) ⇒ *type(c)` (`*` = changed, `~` or `-`); add-type candidates `⇒ +T` | H |
| 7 | `graph_proximity` | graph | `1/(1+d)`, d = undirected shortest path incl. module plumbing (cap 4, else 0) | P + S |
| 8 | `graph_shared_neighbors` | graph | Adamic–Adar: Σ over shared neighbours z of `1/ln deg(z)` (deg floored at 2), divided by the same sum over all seed neighbours | P + S |
| 9 | `graph_cand_degree` | graph | candidate degree / max degree in the snapshot | P |
| 10 | `graph_propagation_conf` | graph | mined relational rule: if c is a direct neighbour, confidence of the cell (seed action ∈ {+,~,−}) × (edge ∈ {in: c references q, out: q references c}), counted over earlier commits; else 0 | H + P + S |
| 11 | `text_name_similarity` | text | Jaccard of name tokens (split on `_ - . /`, camelCase); add-type candidates: seed name vs type name | P + S |
| 12 | `text_content_similarity` | text | TF-IDF cosine (tf = 1+ln count, smoothed idf) of the seed's new body vs the candidate's **parent** body; IDF from the parent snapshot of the same repo | P + S |
| – | `loc_same_file` | locality | 1 if same file — stored, not used by default | P |
| – | `loc_same_dir` | locality | 1 if same directory (module) — stored, not used by default | P |

`model.features` in the config selects the columns the methods use (default: 1–12); changing it needs no re-mine. Add-type candidates: only 6 and 11 are non-zero.

### 7.4 Outputs per repo (`data/processed/<slug>/`)
- `commits.parquet` — `repo, idx, sha, date, author, is_merge, n_items, usable, skip_reason`
- `changes.parquet` — `repo, idx, address, kind, type, action, file`
- `queries.parquet` — `repo, query_id, idx, commit_sha, seed_address, seed_type, seed_action, seed_is_new, n_labels, n_candidates, n_labels_in_candidates`
- `features.parquet` — `repo, query_id, cand_id, cand_kind (existing|add_type), cand_file`, the 14 feature columns of §7.3, `label`
- `parse_errors.parquet`, `_SUCCESS`

**Readable CSV copies.** All tables go through one writer (`processing/io.py::write_table(df, path_stem)`), which always writes `.parquet` (used by the code) and additionally `.csv` when enabled:
- config `output.csv: true|false` (default `true`) and `output.csv_features: true|false` (default `false`, since `features` is by far the largest table; when enabled, write it in full),
- CLI override `tfimpact mine --csv / --no-csv`, plus `tfimpact export-csv [--only <slug>] [--include-features]` to regenerate CSVs from existing parquet at any time.
- List columns (e.g. `refs`) are serialized as `;`-joined strings in CSV. CSV is for human inspection only; code always reads parquet.
- Results tables in `data/results/` are always written as CSV (+ markdown summary).

## 8. Methods

All methods implement `fit(train: list[RepoData]) -> self` and `rank(data: RepoData) -> Ranking` (`methods/base.py`). A `Ranking` holds scores aligned with the candidate rows, an optional `recommended` mask (rows the method actually suggests) and an optional `top_k`; candidates are sorted per query by score, ties broken by a fixed-seed random permutation.

| Method | Description | Learns? | Question it answers |
|---|---|---|---|
| `arm` | ROSE navigation settings: rank existing blocks by `hist_cochange_conf` (ties by `hist_cochange_count`); only rules with support ≥ 1 and confidence ≥ 0.1; top 10; for new seeds the type rule `hist_type_cochange_conf` with the same thresholds | unsupervised (rules) | Better than the prior work it builds on? |
| `fixed_weights` | plain average of the `model.features` columns (GA-IaCRec combines its heuristics with fixed equal weights) | no | Does *learning* the weights add anything? |
| `random` | random scores (chance-level reference line, not a competitor) | — | — |
| **`ga`** | **self-implemented GA learning w over the `model.features` columns** | **yes (main)** | |

The single-feature rankings are reported as a univariate analysis, not as baselines (decision §15.13).
| `gp` (stretch) | self-implemented GP evolving the scoring expression | yes |
| `lgbm` (stretch) | LightGBM LambdaRank on the same features | yes |

### 8.1 GA specification (`methods/ga/`, no external GA library)
- **Chromosome:** 12 weights w ∈ [0,1]^F (one per `model.features` column), normalized to sum 1 (scale invariance; each weight reads as "share of trust in this signal"), plus one **add-type offset** b ∈ [−1, 1] (a bias term for add-type candidates, which have only two non-zero features; since scores lie in [0, 1], ±1 can place them anywhere in the list). Non-negative weights by default; one check run allows negative weights (normalized by Σ|w|) to confirm that no feature acts as a suppressor.
- **Score:** `X @ w` for existing-block candidates, `X @ w + b` for add-type candidates.
- **Fitness (single objective):** MRR on **all targets** over a fixed sample of training queries (other metrics are reported in evaluation; multi-objective optimisation is future work). The sample has an **equal number of queries per repo** (repos weighted equally, as in evaluation), is fixed within a run and drawn anew per seed. Queries with no true answer among their candidates are left out (their RR is 0 for every chromosome); they still count in evaluation. Ties use the expected rank under random tie-breaking. Fully vectorized (precompute X, y, group offsets; evaluate the whole population with one matrix product). Subsampling computes the *real* metric on fewer queries (an unbiased estimate of the training MRR, compared on the same queries for all individuals of a run); no surrogate model, since one evaluation costs ~ms.
- **Operators:** tournament selection (size 3), blend crossover (BLX-0.5, rate 0.8), Gaussian mutation (rate 0.3 per gene, σ linearly decaying 0.2 → 0.02), clip + renormalize, elitism (2).
- **Defaults:** population 60, generations 60, seeds 10 per fold (31 if affordable, decided after timing). Log best fitness per generation (convergence plot).
- **Settings are justified, not assumed** (milestone 5): (1) **sample-size check** on one fold (2,000 / 5,000 / all training queries, 3 seeds each; §15.14); (2) convergence check with a generous budget fixes population × generations; (3) nested grid search per fold over population/generations at that budget (3 levels), mutation rate (0.1 / 0.3 / 0.5) and crossover rate (0.6 / 0.8 / 0.95): each configuration is trained on 75 % of the fold's training repos and scored on the other 25 % (`splits.inner_split`), the best one is retrained on 100 % of the training repos and only then tested. Tournament size 3 and elitism 2 stay fixed. Results recorded in `docs/THRESHOLDS.md`.
- **Label-permutation check:** one run per fold with labels shuffled within each training query; test MRR must collapse (evidence that the gain comes from learning, not leakage).
- **Random search** with the same evaluation budget is a method of its own (`random_search`): in the main table and the statistical tests (does the GA's search beat sampling weights at random?).
- **Resumable runs:** every finished run (configuration, fold, seed) writes its result (weights, offset, per-generation fitness, settings hash) to `data/results/ga_runs/<config>/fold<k>_seed<s>.json`; reruns skip finished runs, so tuning and training can be interrupted and resumed with the same command. Each run seeds its own RNG from (base seed, configuration, fold, seed), so interrupted and uninterrupted executions give identical results. A changed GA setting changes the hash and re-runs.
- **Final model** (`tfimpact train`): one run on all 53 core repos with the configuration the grid search chose most often across folds; written to `models/ga_weights.json` (features, weights, offset, configuration, feature settings) and used by `recommend`. It is never scored on these repos (its expected quality is the cross-validation estimate); case studies use repos outside the dataset or the fold model that did not see the repo.
- Output: weights JSON + convergence history.

### 8.2 GP (stretch, `methods/gp.py`)
Terminals = features + ephemeral constants; functions = add, sub, mul, protected div, max, min, log1p; ramped half-and-half init (depth 2–5); subtree crossover/mutation; tournament + elitism; max depth 7; fitness = MRR − 0.002·size; vectorized evaluation. Report the best expression. Optional one-off comparison with DEAP gp.

## 9. Evaluation

### 9.1 Metrics (`evaluation/metrics.py`)
Per query: reciprocal rank (first hit; MRR is the primary metric and the GA fitness), average precision (MAP: rewards ranking *all* true answers high), Recall@k, Precision@k, Hit@k (= GA-IaCRec's success rate) for k ∈ {1, 3, 5, 10} (3 matches ROSE and GA-IaCRec). Recall and AP divide by *all* true answers, so answers lost by the candidate cap count as misses. Rows outside a method's `recommended` mask or `top_k` never count as found. Aggregate per repo = mean over its queries; overall = mean over repos (each repo weighted equally). Reported for two target sets: **all** (existing blocks + add-type candidates) and **existing** (existing blocks only, stricter). Also:
- SD over repos, SD over folds (and over GA seeds), **bootstrap 95% CI** of the mean over repos (1,000 resamples),
- candidate-recall ceiling,
- breakdowns: seed action (added / modified / removed) and repo kind,
- **file-level aggregation** (map ranked blocks to files, keep first occurrence) as one supplementary line for GA-IaCRec's granularity.
- No "lenient" future-commit labels (decision §15.13).

### 9.2 Splits (`evaluation/splits.py`)
- **Grouped 5-fold cross-validation (the only protocol):** **by repository** (each repo is tested exactly once, on weights learned from the other folds). Folds are balanced by non-warm-up query count (seeded shuffle, largest-first greedy assignment) and saved to `data/results/folds.csv` so every method uses identical folds. Features are always per repo and causal; only learned parameters (GA weights) cross repos. GA: 10 seeds per fold → 50 runs.
- **Warm-up queries** are excluded from training and evaluation (they still fed the miners).
- **Add-only repos held out:** the 8 repos excluded only for `max_add_only_frac` are never trained on; every run also evaluates them (learned methods: with each fold's model, averaged over folds) and reports three subsets: **core** (53, headline; statistics, breakdowns and file level use only these), **add_only** (8) and **all** (61).
- P2 (chronological 90/10) was dropped: its only purpose was comparability with GA-IaCRec, whose numbers are not comparable anyway (files of Ansible/Puppet projects vs Terraform blocks).

### 9.3 Statistics (`evaluation/stats.py`)
Pairwise on **per-repo** values (n = number of repos; queries of one repo/commit are not independent), for MRR and MAP and both target sets: Wilcoxon signed-rank (two-sided), Vargha–Delaney A12 with magnitude (negligible / small / medium / large at 0.56 / 0.64 / 0.71), Holm correction within each (target, metric) family. GA per-repo values = mean over its seeds.

### 9.4 Experiments (`evaluation/experiments.py`)
1. **Main comparison** — `ga` vs `arm` / `fixed_weights` (+ `random` line), in one cross-validation run: headline on the 53 core repos, plus the 8 add-only repos as a held-out group (never trained on) and all 61 together.
2. **Ablation** — single-family variants GA-HIST / GA-GRAPH / GA-TEXT (prefixes `hist_`, `graph_`, `text_`; GA-IaCRec's RQ2 design adapted to learned weights) vs the full GA (decision §15.17).
3. **Learning curve** — GA trained on 10/25/50/75/100 % of each fold's training **repos** (all their queries); test MRR vs number of training repos (Mitchell's T/E/P argument; decision §15.17).
4. **Learned weights** — table/plot of mean GA weights ± std across runs; top mined type-level rules.
5. (stretch) cutoff sensitivity `max_items` ∈ {20, 30, 50}.

### 9.5 Real-world validation
- **Live test** (required): `recommend --diff` on prepared edits in repos outside the dataset, expected answers written beforehand, 2–3 raters; procedure and recording in [`LIVE_TEST.md`](LIVE_TEST.md).
- **External set** (optional, planned, not implemented; decision §15.17): the final model scored on every qualifying commit of new TerraDS repos (star ranks 151–200).

Outputs in `data/results/`: `metrics.csv` (long format: method, target, subset, metric, mean, SDs, CI), `per_repo.csv`, `breakdowns.csv`, `file_level.csv`, `stats.csv`, `per_query.parquet`, `learned_weights.csv` (one row per GA/random-search run), `run.json` (config, seed, folds, method parameters, code version), `summary.md`, `folds.csv`, `ga_runs/` (resumable per-run JSON), `convergence.csv` / `convergence_summary.csv` / `convergence.png`, `checks/` (sanity variants via `--out checks`), `ablation.csv`, `learning_curve.csv`, `ga_weights.json`, plots (`convergence.png`, `learning_curve.png`, `weights.png`), `results.md` summary.

## 10. CLI (`tfimpact`)

```
tfimpact select-repos --terrads data/external/terrads/dataset.sqlite [--limit 150]
tfimpact mine [--repos data/raw/repos.csv] [--only <slug>] [--force]
tfimpact clean
tfimpact export-csv [--only <slug>] [--include-features]
tfimpact evaluate [--methods ga,random_search,arm,fixed_weights,random] [--out <name>]   # trains per fold (incl. grid search)
tfimpact train                      # final model on all core repos -> models/ga_weights.json
tfimpact experiments --convergence [--ablation] [--learning-curve]
tfimpact export [--name main]       # copy result files + figures -> results/<name>/ (committed)
tfimpact report                     # (milestone 7) written results document
tfimpact recommend --repo PATH (--block <address-or-type.name> | --diff) [--top 10] [--explain]
```

`recommend` (`inference/recommend.py`, implemented 2026-11-17):
- `tfimpact recommend --repo PATH (--block ADDRESS | --diff) [--top 10] [--explain] [--model PATH] [--json]`.
- Mines the local repo's history up to HEAD with the training pipeline (`process_repo(..., compute_features=False)` returns the final `RepoState`: snapshot, graph, token/content indexes, rule miners), last `mining.max_commits` commits; no clone, no network, no Terraform tooling.
- Seeds: `--block` = one HEAD block (full address or short form such as `aws_s3_bucket.logs`, `var.x`; ambiguous names are rejected), treated as modified with its current body; `--diff` = every block that differs between the working tree and HEAD (`git diff HEAD` + untracked `.tf` files; deleted files remove their blocks; a rename appears as remove + add), with the edited bodies. Unparsable edited files are reported and kept at their HEAD version.
- Features from the training `FeatureBuilder` at index HEAD + 1 (verified: identical to the pipeline's features for the same edits committed, `tests/test_recommend.py`); score = model weights · features (+ offset for add-type). Several seeds: a candidate keeps its best score (`recommend.combine: max`; `sum` available). Seeds themselves are never suggested.
- Output: two lists, "Blocks to modify" and "Block types to add" (`recommend.top` each); `--explain` adds the top `recommend.explain_top` contributions (weight × feature); `--json` for scripts and case studies.
- Model: `models/ga_weights.json` by default; refused if its feature list is unknown to the code; its stored `features_config` is applied so features match training.

## 11. Configuration (`configs/default.yaml`, sketch)

```yaml
paths: {terrads_db: data/external/terrads/dataset.sqlite, repos_csv: data/raw/repos.csv,
        tmp: data/tmp, processed: data/processed, results: data/results}
selection: {provider: aws, min_resources: 20, shortlist: 150, min_usable_commits: 50, target_repos: 25}
mining: {max_commits: 2000, max_items: 30, min_items: 2, warmup_frac: 0.2}
features: {max_candidates: 200, graph_max_dist: 4, recency_tau: 100, churn_window: 200}
ga: {population: 60, generations: 60, tournament: 3, crossover_rate: 0.8, mutation_rate: 0.3,
     sigma_start: 0.2, sigma_end: 0.02, elitism: 2, fitness_sample: 2000}
evaluation: {k: [1, 5, 10], folds: 5, seeds: 10, temporal_test_frac: 0.1}
output: {csv: true, csv_features: false}
seed: 42
```

## 12. Testing
- Parser: resources/data/modules/variables/outputs/locals, nested blocks, interpolations, parse error handling.
- References: resource, data, module, var, local, depends_on; no cross-directory leakage.
- Diff: add/remove/modify; formatting-only change ⇒ no item.
- **Causality test:** on a tiny synthetic git repo, assert no feature for commit t uses counts from t or later.
- Metrics: hand-computed RR/R@k/P@k.
- GA: improves fitness on a toy problem; deterministic for a fixed seed.

## 13. Milestones (≈ 7 × 2–3 h)

| Day | Deliverable | Done when |
|---|---|---|
| 1 | uv skeleton, config, CLI stub; `select-repos`; `bare_clone`; parser + diff on **one** repo | `changes.parquet` for one repo looks sane |
| 2 | snapshot walker, filters, warm-up; run `mine` on ~20 repos with clean-up | `summary.csv` with ≥ 20 repos, parse rate reported, `data/tmp` empty |
| 3 | rules (instance + type), graph, similarity, candidates, feature table | `features.parquet` per repo; causality test passes |
| 4 | baselines (`arm`, `fixed_weights`), metrics, grouped k-fold splits, stats, `evaluate` | baseline metrics tables (done 2026-11-17) |
| 5 | own GA + random search; convergence check + nested grid tuning; main comparison | `metrics.csv`, `stats.csv`, convergence plot (done 2026-11-17; `results/main/`, `models/ga_weights.json`) |
| 6 | `recommend` CLI; ablation; learning curve | live demo works on a repo; ablation + curve tables (done 2026-11-18; `results/ablation/`, `results/learning_curve/`) |
| 7 | `report`, case studies, README; buffer | `results.md` complete |

Stretch work (GP, LightGBM, provider-neutral categories) only after Day 5 results exist.

## 14. Known risks & mitigations
- **HCL parse failures** → log & keep previous blocks; report parse rate; fallback `hcl2json` binary if rate < 90%.
- **Too few usable repos** → relaxation ladder (5.1).
- **Feature table too large** → candidate cap, float32, per-repo parquet, GA fitness sampling.
- **Tangled commits** → `max_items` cutoff, merges excluded.
- **Candidate pre-filter hides labels** → report candidate-recall ceiling.
- **Threats to validity to discuss in report:** commit co-change ≠ necessary change; AWS-only sample; module libraries vs deployment repos; renames treated as remove+add.

## 15. Implementation notes

Findings from inspecting the real libraries/data (2026-11-08). Where these differ from earlier sections, these notes win. **All selection/mining thresholds and their justifications live in [`THRESHOLDS.md`](THRESHOLDS.md).**

### 15.1 TerraDS (`data/external/terrads/TerraDS.sqlite`, 416 MB)
- File name is `TerraDS.sqlite` (not `dataset.sqlite`); path is in `configs/default.yaml`.
- Tables (plural): `Repositories` (62,406 rows), `Modules` (279,344), `Resources` (1,773,991).
  - `Repositories`: `Id, Name, FullName, CreatedAt, Description, License, StarCount, ForkCount, HtmlUrl, GitUrl, SshUrl, CloneUrl, Homepage, Archived (0/1), Topics (JSON list), LatestCommitAt, LatestCommitSha, SizeInKb`.
  - `Modules`: `Id, RepositoryId, Path, Providers (JSON list), ModuleCalls (JSON list), DiagnosticMessages`. A "module" = a directory.
  - `Resources`: `Id, ModuleId, ResourceType ∈ {'Managed','Data'}, Name, Type, Provider` (e.g. `aws`, `azurerm`, `google`).
- Join path: `Resources.ModuleId → Modules.Id`, `Modules.RepositoryId → Repositories.Id`. Provider filter = `Resources.Provider = :provider`.
- Snapshot ends 2024-09-02 (`LatestCommitAt`); we clone current HEAD, so histories extend past TerraDS — fine, but some repos may be deleted/renamed (handle clone failures).
- 9,296 non-archived repos have ≥ 20 AWS managed resources (682 with ≥ 10 stars, 213 with ≥ 50) — selection pool is ample.

### 15.2 python-hcl2 (v8.1.4) output format
- Must call `hcl2.loads(text, serialization_options=SerializationOptions(with_comments=False, explicit_blocks=False, preserve_heredocs=False, strip_string_quotes=True))`. Defaults in v8 differ from older docs: labels come quoted (`"\"aws_s3_bucket\""`), `__is_block__` markers and a top-level `__comments__` key are emitted.
- With those options the shape is: top-level key per block keyword → **list** of dicts.
  - `resource` / `data`: `[{type: {name: body}}]`; `module` / `variable` / `output` / `provider`: `[{name: body}]`; `locals`: `[{key: expr, ...}]` (one dict per `locals {}` block — merge them); `terraform`, `moved`, `import`, `check`, `removed`: `[body]` (ignored as blocks).
  - Nested blocks (`statement`, `validation`, `dynamic`) are lists of dicts; attributes are scalars / lists / dicts.
  - Expressions are strings `"${...}"`, **re-serialized canonically** (whitespace normalized: `var.create?1:0` → `${var.create ? 1 : 0}`); string templates stay as-is (`"${local.a}-suffix"`); heredocs become plain strings; `depends_on` is a list of `"${addr}"`.
- Verified: comment-only and whitespace/alignment-only edits produce **identical** output for every block kind → `body_hash` over canonical JSON drops formatting-only changes as SPEC §6.4 intends.
- Speed: ~6 ms/file on real repos (162 files, 0 failures on `terraform-aws-vpc` + `terraform-aws-secure-baseline`). Parse results are cached by **blob SHA**, so each file version is parsed once.

### 15.3 Git access
- **Clone mode `blobless_prefetch` (default):** `git clone --bare --filter=blob:none`, then compute the exact set of `*.tf` blob OIDs needed (`ls-tree -r` at the start commit + `log --first-parent -m --raw --no-abbrev -- '*.tf'`) and fetch them in chunks with `git fetch origin <oids…>`. Verified on GitHub: `modernisation-platform` → 3,075 blobs in 3 s, 24 MB (full bare clone: 3 s, 43 MB). Scales to very large repos (e.g. `kubernetes/k8s.io` ~1 GB) without downloading non-Terraform content. `mode: full` remains as a fallback.
- **Depth check** uses the blobless clone only (no blobs needed): count first-parent commits touching `*.tf`. Exact *usable* counts (2–30 changed blocks) need parsing, so they are measured during mining and repos below `min_usable_commits` are dropped afterwards (two-stage selection: proxy `min_tf_commits`, then exact).

### 15.4 Merge-heavy repositories (open decision)
- Depth scan of the top-40 candidates: module libraries (`terraform-aws-modules/*`, `cloudposse/*`, `poseidon/typhoon`) are squash/linear — non-merge commits dominate. But most **deployment** repos use merge-commit PR workflows: `alphagov/govuk-aws` 1 non-merge vs 1,313 merge `.tf` commits; `modernisation-platform` 657 vs ≥ 2,000; `kubernetes/k8s.io` 33 vs 570; `kbst/terraform-kubestack` 4 vs 183.
- Under §6.5 (merges never transactions) these repos contribute ~nothing, biasing the sample toward module libraries.
- **Decision (2026-11-08): `mining.use_merge_commits: true`.** A first-parent merge commit is one PR-level transaction, diffed against its first parent (same granularity as a squash merge); `max_items` still drops oversized PRs. `commits.is_merge` is kept so a with/without-merges sensitivity check can be reported. This **supersedes §6.5's "not a merge" rule and §6.1's "never use as transactions"**. PoC: `modernisation-platform` → 285 usable commits, all merges (would be 0 otherwise).

### 15.5 Repo kinds and the add-only ("accretion") signal
- Selection keeps **everything passing the depth check** (user decision); `repo_kind` ∈ {module_library, deployment, educational} is a heuristic tag (name/description/topics) used only for breakdowns.
- Risk: educational repos (e.g. `antonputra/tutorials`, `ned1313/terraform-tuesdays`) grow by adding self-contained lesson folders — commits are mostly `+` items, seeds are mostly new (cold start), labels mostly add-types. Under per-repo-equal weighting this pulls GA weights toward `type_conf`. Mitigation is **data-driven**: `summary.csv` records `add_only_frac` (share of usable commits consisting only of `+` items) and `add_item_frac`; repos with extreme accretion can be excluded with a stated threshold, and results reported with/without them.
- **Decision (2026-11-08): `selection.max_add_only_frac = 0.4`.**
  - *Why exclude:* in an add-only commit the seed is a brand-new block, so the history features f1–f3 and f12 are 0, and every positive label is an add-type candidate that only f4 (`type_conf`) can score. A repo made almost entirely of such commits measures "which blocks are written together", not change impact. Because CV weights repos equally, each such repo biases the learned weights toward f4 and distorts test-fold scores.
  - *Why 0.4:* run 1 (40 repos) was clearly bimodal. 34 repos fall in [0.02, 0.32], then there is a gap, then 6 repos fall in [0.45, 0.96]: `hashicorp/terraform-provider-awscc` 0.96 (generated doc examples), `antonputra/tutorials` 0.93, `devopsacademyau/academy` 0.88, `ned1313/terraform-tuesdays` 0.65, `stelligent/config-lint` 0.53 (linter fixtures), `ManagedKube/kubernetes-ops` 0.45. Any threshold in [0.33, 0.44] gives the same split; 0.4 is the round value inside the gap.
  - *Threat to validity:* the threshold was chosen after seeing the data. The defence is the two-cluster gap plus reporting the main results with and without these repos. Only whole repos are dropped; add-only commits inside kept repos remain (add-type prediction is a contribution).

### 15.7 Repo funnel and the `build-dataset` step
- Funnel: TerraDS (≥ 20 managed resources of `provider`, **archived repos included** — decision 2026-11-08) → top `shortlist` (150) by stars, or all with `select-repos --all` → depth proxy (≥ `min_tf_commits` first-parent `.tf` commits; blobless, no parsing) → **fork dedup** (same root commit ⇒ keep most-starred) → **mine** → **build-dataset** (≥ `min_usable_commits` usable commits **and** `add_only_frac` ≤ `max_add_only_frac`) → `data/raw/dataset.csv` (`included`, `exclude_reason`). Downstream stages read only repos with `included = true`.
- `build-dataset` must follow mining: "usable" (2–30 changed **blocks**) and `add_only_frac` need every file version parsed, which is what mining does. It runs automatically at the end of `tfimpact mine`; `tfimpact build-dataset` re-applies changed thresholds in seconds without re-mining.
- **Pinning:** `select-repos` records each repo's `head_sha` in `repos.csv`; `mine` walks history from that SHA (re-running reproduces identical tables) and skips repos already mined at it. `--latest` (on `select-repos` and `mine`) uses the remote's current HEAD instead, re-mines only repos that have new commits, and updates the pins. The SHA actually mined is stored in `meta.json`.

### 15.6 Pipeline details that refine §6
- **No rename detection at all** (`git log --no-renames`): a moved file within the same directory keeps all block addresses (address = dir + block id, not file), so a pure file move/rename yields no items. Moves across directories are remove + add, as in §6.4.
- **Diff is per affected directory**, not per file: before/after = all blocks of every directory containing a touched `.tf` file. This handles blocks moved between files and duplicate addresses correctly.
- Snapshot update is two-phase (`stage` → [queries] → `apply`), which makes the leakage boundary in §6.6 explicit in code.
- **Warm-up applied post hoc:** queries are emitted for every usable commit and the first `warmup_frac` of usable commits are flagged (`commits.in_warmup`) and dropped. Since features at t only use data < t, this equals not emitting them, and avoids needing the usable count before the walk.
- Commits walk: one `git log --first-parent --diff-merges=first-parent --raw --no-abbrev --no-renames -n max_commits` call; symlinked / submodule `.tf` entries are ignored.
- Parse failures seen so far are genuine (e.g. committed merge-conflict markers `<<<<<<< HEAD`); parse rate 99.8–99.9 % on PoC repos.
- PoC timings: `terraform-aws-vpc` (495 commits, 134 usable) 24 s; `modernisation-platform` (2,000 commits, 285 usable) 36 s — parsing dominates; blob prefetch < 1 s.

### 15.8 Milestone-3 decisions (2026-11-08, confirmed by the author)
- **Miner input:** the RuleMiner is updated with every commit that has **1–30 items** (single-block commits included, tangled > `max_items` commits excluded; merges per `use_merge_commits`). Single-block commits correctly lower `N(q,c)/N(q)`. Only usable commits (2–30) produce queries.
- **Seeds:** every changed block is a seed: `+`, `~` and `-`. `seed_action` and `seed_is_new` are stored for breakdowns (§9.1).
- **f4 `type_conf` for existing candidates:** the target side means "type T **changed**" (`~` or `-` merged), matching the label definition in §7.2. For add-type candidates the target stays `+T`.
- **Module plumbing edges (§6.3 nice-to-have) are core:** a `module` block with a local `source` (`./`, `../`) gets edges to the variables it sets and the outputs it reads in the resolved module directory. Remote sources (registry, git) get no edges.
- **Seed body in graph features:** f5, f6 and f7 all use the seed's **new** body (its references as just written), so the three graph features are mutually consistent. No other post-commit information is used.
- **Repo scope for features:** the 53 included repos + the 8 repos excluded only by `max_add_only_frac` (evaluated as a held-out group, §9.2).

### 15.9 Alignment with ROSE (verified against `docs/2004_rose-version-history-mining.pdf`)
Zimmermann, Weißgerber, Diehl, Zeller, *Mining Version Histories to Guide Software Changes*, ICSE 2004.
- **Same:** transactions = changes committed together (§3); rules `Σ ⇒ {e}` with support and confidence (§4); **Apriori with constrained antecedents and single consequents, mined incrementally on the fly** (§4.2). With a single-entity antecedent and single consequent this reduces to exact pair counting, which is what `AssociationRuleMiner` does. Ranking by confidence (§4.1). **Transactions affecting > 30 entities ignored** (§3; our `max_items`). Evaluation replays history: for each transaction and **each entity e, query Q = {e}, expected E = entities(Δ) − {e}**, rules mined only from transactions completed before Δ (§6.1–6.2). This is exactly our seed/label construction and leakage rule.
- **Metrics:** ROSE reports precision, recall and L3 (≥ 1 correct in the top 3, our Hit@3 analogue); navigation results use min support 1, min confidence 0.1, top-10 by confidence (§6.4). ROSE notes that projects adding many new features are less predictable from history (§6.4), the motivation for our type-level rules / add-type predictions.
- **Differences (ours):** git commits instead of CVS sliding-window reconstruction (200 s window); first-parent merges used as PR-level transactions, where ROSE avoided merges (they were huge CVS merges; our `max_items` still drops large ones); no min-support/confidence thresholds in the *features*; Terraform blocks + type-level rules; learned multi-feature ranking; cross-repository evaluation.

### 15.10 `arm` baseline settings (decision 2026-11-08)
- Parameters shared with ROSE use **ROSE's original navigation settings** (§6.4 of the paper): rank by confidence, **min support 1**, **min confidence 0.1**, only the **top 10** suggestions are returned. Candidates below the thresholds or outside the top 10 count as not recommended (score 0), which lowers MRR relative to a full ranking.
- Our own extensions are kept: **fallback to type-level confidence (f4) when the seed is new** (ROSE returns nothing there), and ties broken by support (f2), then a seeded random.
- `top_k` (config `baselines.arm.top_k`) stays **open**: if the dev check shows 10 is too small, it may be raised, with the change recorded here.

### 15.11 Evaluation & reporting decisions (2026-11-08)
- **Labels = same transaction only** (as ROSE): changes in later commits are never features *and* never labels. Rationale: features must not see the future (deployability); for labels, many transactions are already PR-level (merge / squash commits) so in-PR follow-ups are included, and missed follow-ups make results pessimistic, not optimistic. *Stretch:* label-window sensitivity (also count blocks changed by the same author within the next 24 h as positives; labels only).
- **Two headline columns for every method:** "all targets" and "existing-block targets", plus breakdowns (seed new vs existing; add-type targets). Generic add-type targets (`+variable`, `+output`, `+local`, `+module`) are valid but easy (35–65 % of add-type positives on the dev repos), so the existing-block column is the stricter measure. If the GA over-weights f4 because of them (visible in weights + ablation), training on existing-block targets is to be discussed.
- **`recommend` output shows two separate lists:** "blocks to modify" (existing) and "block types to add", so add-type suggestions never displace existing-block ones.
- **Fairness:** features/rules are per-repo, causal memory (what a deployed tool would mine from the user's repo); the only cross-repo learned parameters (GA weights) are evaluated on unseen repos (grouped cross-validation). Case studies use fold weights that did not see the repo, or repos outside the dataset; "all-data" weights are used only by the shipped `recommend` tool.
- **Re-mining guard:** `meta.json` stores `pipeline_version` and `settings_hash` (mining + features config); `mine` re-processes a repo when either differs, and `build-dataset` ignores stale outputs. Adding repos never re-processes existing ones; changing a setting always does. `PIPELINE_VERSION` lives in `constants.py` (it describes the code, not a tunable) and is bumped for any code change that alters the per-repo tables; `meta.json` also lists the stored feature columns.

### 15.12 Feature set v2 (decisions 2026-11-16, confirmed by the author; `PIPELINE_VERSION` 3)
Earlier sections (§15.8–15.10) use the v1 names; mapping: f1 `inst_conf` → `hist_cochange_conf`, f2 `inst_support` → `hist_cochange_count`, f3 `recency` → `hist_cochange_recency`, f4 `type_conf` → `hist_type_cochange_conf`, f6 `graph_prox` → `graph_proximity`, f7 `nbr_jaccard` → `graph_shared_neighbors`, f8 `cand_degree` → `graph_cand_degree`, f9 `name_sim` → `text_name_similarity`, f10/f11 → `loc_same_file` / `loc_same_dir`, f12 `cand_churn` → `hist_cand_recent_changes`. Evidence below is from the v1 feature tables (53 included repos, non-warm-up queries).
- **Names** carry the family as prefix (`hist_`, `graph_`, `text_`, `loc_`), used to select ablation families. Type rules are mined from history, so they belong to `hist_`.
- **Dropped `ref_direct`:** on every existing-candidate row, `ref_direct = 1` exactly when `graph_proximity = 0.5`, so it carried no information of its own. It only let a linear score boost distance 1 over distance 2, which the data does not support: P(true answer) is 10.8 % at d = 1 and 11.8 % at d = 2 (28 % of true answers sit at d = 2, 8 % at d = 1), and drops to 4.5 % at d = 3.
- **Added `hist_file_cochange_conf`** (GA-IaCRec's history heuristic at file granularity, ROSE's coarse granularity): only 27 % of true existing answers have block-level co-change history, 72 % lie in another file than the seed, and 30 % of those cross-file answers have file-level but no block-level history. It also gives evidence for new seeds (39 % of seeds), whose file usually has history.
- **Added `graph_propagation_conf`** (a mined relational rule over reference edges, preferred to a hard-coded "removed ⇒ referencers change" rule): direct neighbours of a **removed** seed are true answers 58.5 % of the time vs 6.5 % (modified) and 9.0 % (added) seeds. A linear score cannot condition `graph_proximity` on the seed action, so the interaction needs its own feature. Direct neighbours only; 6 cells (action × in/out); a neighbour linked both ways counts as `in`; updated with the same commits as the rule miner.
- **Added `text_content_similarity`** (GA-IaCRec's content heuristic at block granularity): links copies and blocks sharing settings that have no reference path, e.g. typhoon's per-platform copies (88 % of its true targets unreachable in the graph). Leakage-safe construction: candidates use their **parent** bodies (the post-commit body of a co-edited copy would encode the label), IDF counts are kept incrementally over the **parent** snapshot of the same repo (not fitted on a later snapshot or across repos). Verified by an independent recomputation test and by injecting the leak.
- **`graph_shared_neighbors` → Adamic–Adar** instead of Jaccard: shared hubs (`local.common_tags`, `var.region`) count little, shared rare neighbours a lot; normalised by the seed's own maximum so the value is in [0, 1] without depending on the other candidates. No extra cost (degrees are computed per commit anyway).
- **Locality stored but off by default:** a proxy ablation (random weight search, 5-fold by repo, 3 seeds) showed no measurable effect of removing `loc_same_file` + `loc_same_dir` (MRR difference −0.010 to +0.016, smaller than the 0.027 spread between seeds). They are still computed (free) so the decision stays reversible via `model.features` and can be reported as an ablation line.
- **Kept `graph_cand_degree`** despite being the weakest single discriminator (non-zero for 97 % vs 98 % of true/false candidates; MRR 0.254 vs random 0.119): free to compute, the structural counterpart of the churn prior; the GA weights and ablation decide.
- **Unchanged:** candidate pre-filter and `max_candidates` (configurable), labels, query construction, all thresholds of §15.6–15.8.
- **Future work (not implemented):** deterministic Terraform validity rules (output/variable removal), lift, directory-level rules, multi-item antecedents (TARMAQ), body-size/complexity metrics (planned as an evaluation breakdown instead).

### 15.13 Milestone-4 decisions (2026-11-17, confirmed by the author)
- **Baselines reduced to two** (+ a chance line), each answering one question a reader asks (course requirement: evidence that the result is better than prior work, Shaw 2003): `arm` (prior work: ROSE) and `fixed_weights` (does learning matter?). `cochange`, `references`, `type_prior` dropped; the per-feature table (§15.12) covers single signals. `arm_borda` (GA-IaCRec's ARM) skipped.
- **`arm` and min support:** stored features keep confidence and a per-query normalised support, so only ROSE's "support ≥ 1" (= any co-change, confidence > 0) can be applied exactly; other values are rejected. Thresholds are enforced through the `recommended` mask (rules below them are never suggested), the top-10 limit through `top_k`.
- **Cross-validation only** (no protocol tag in result file names), folds balanced by query count; file named `splits.py`. **Warm-up** excluded from training and evaluation.
- **Add-only repos:** no separate sensitivity run (`--with-accretion` removed). They are never trained on and always evaluated as a held-out group (external-validation style), reported next to, never mixed into, the core headline; "all 61" gives the with-them number. Training *with* them is not tested by default (a one-off experiment if ever needed).
- **Metrics:** MAP added, k = {1, 3, 5, 10}, SDs + bootstrap CI, breakdowns by seed action and repo kind, file-level kept as one supplementary line. **No lenient future-commit labels:** "changed in any later commit" makes almost every suggestion correct eventually, and any time window is arbitrary.
- **Statistics** on per-repo values (Wilcoxon + A12 + Holm), the standard recipe for randomized algorithms in SE (Arcuri & Briand, ICSE 2011).
- **GA settings** are tuned by convergence check + nested grid search on training repos only (§8.1); no setting is picked from test results.
- **Baseline results** (53 repos, `data/results/summary.md`): MRR all / existing — `arm` 0.481 / 0.360, `fixed_weights` 0.563 / 0.644, `random` 0.118 / 0.116; `fixed_weights` beats `arm` on 39 of 53 repos (Holm p = 0.0015, A12 = 0.66 medium). By seed action (all targets): `arm` is better for **added** seeds (0.507 vs 0.388, type-rule fallback, while equal weights dilute the two non-zero add-type features), `fixed_weights` for modified (0.640 vs 0.414) and removed seeds (0.774 vs 0.335). No difference by repo kind. Held-out add-only repos (MRR all / existing): `arm` 0.667 / 0.268, `fixed_weights` 0.177 / 0.630; all 61: `fixed_weights` 0.513 / 0.642. Full record: `docs/RESULTS.md`. SD over repos ≈ 0.15, over folds ≈ 0.04–0.06. File level: `fixed_weights` 0.768, `random` 0.338 (file level is much easier, which is why it stays supplementary).

### 15.14 Milestone-5 decisions (2026-11-17, confirmed by the author)
- **Fitness = MRR, all targets, single objective;** weighted-sum or multi-objective fitness rejected (extra arbitrary weights or a Pareto set to choose from).
- **Add-type offset gene** b ∈ [−1, 1]: a learned bias for add-type candidates, because one weight vector must place existing blocks (12 features) and add-type candidates (2 non-zero features) in the same list; equal weights fail exactly there (new seeds, add-only repos, `docs/RESULTS.md` §3.4).
- **Non-negative weights** (all features were designed and measured as "higher = more related", `docs/RESULTS.md` §2.4), checked once with negatives allowed.
- **Fitness sample:** equal number of queries per repo, fixed per run; size decided by a one-fold check (2,000 / 5,000 / all). GA-IaCRec has no such choice: its GA searches one file list per test query and its fitness uses only heuristic scores, never labels, so there is no training set to sample. Subsampling is fitness approximation by a cheaper version of the same problem (not a surrogate model; Jin, Soft Computing 2005; training-subset selection in GP, Gathercole & Ross 1994).
- **13 genes kept** (12 shared weights + one add-type offset): a 15-gene variant with separate add-type weights for the type rule and name similarity gave +0.002 MRR, within seed noise (`docs/RESULTS.md` §4.1). **Fitness sample = 5,000 queries** (matches all-data test MRR at 1/9 of the cost, §4). **10 seeds per fold** (seed spread ≈ 0.0004 MRR).
- Queries without a true answer among the candidates are excluded from the fitness only; `random_search` is a full method; grid as in §8.1; seeds per fold after timing; resumable per-run result files; demo weights trained on all core repos.
- **Stage 1 = one command** (2026-11-17): `evaluate` trains `ga` *with* the nested grid search per fold (27 configurations × 3 seeds on 75 % of the fold's training repos, scored on the other 25 %), then 10 seeds with the chosen configuration; `random_search`, `arm`, `fixed_weights`, `random` on the same folds. No separate `ga_default` method: the default configuration is a grid point, so `grid.csv` shows tuned vs default on validation. Sanity variants (`ga_permuted`, `ga_negative`) run separately into `results/checks/`. Figures: learned weights, per-repo MRR box plot, MRR by seed action (`results/figures/`).

### 15.15 Artifact locations (2026-11-17)
- Following the Cookiecutter Data Science convention: trained model in **`models/`** (`ga_weights.json`, 2 KB, committed so a clone can run `recommend` on another machine), committed result tables and figures in **`results/<experiment>/`** (`main/`, later `ablation/`, `learning_curve/`; written by `tfimpact export --name <experiment>`), full and large outputs (per-query parquet, `ga_runs/`) stay in the gitignored `data/results/`. Written documentation only in `docs/`. Config keys: `paths.models` (`models/`), `paths.results` (`results/`, committed) and `paths.outputs` (`data/results/`, everything runs write).

### 15.16 `recommend` decisions (2026-11-17)
- Several `--diff` seeds are combined by **max** (a candidate related to any edit ranks high); history depth = `mining.max_commits` (2,000), the conditions the model was evaluated under. Both configurable (`recommend.combine`, `mining.max_commits`).
- Propagation rules are now updated in every mining pass (also without query features), so the state `recommend` mines is complete; per-repo tables are unchanged.
- History limit `mining.max_commits` = 2,000 is kept for `recommend`: the model was evaluated under it (only 5 of 53 core repos reach it), each call re-mines, and a longer history is an untested condition. Named as a threat to validity.

### 15.17 Milestone-6/7 decisions (2026-11-18, confirmed by the author)
- **Ablation:** three single-family variants only, GA-HIST (6 `hist_` features), GA-GRAPH (4 `graph_`), GA-TEXT (2 `text_`), against the full GA. Same design question as GA-IaCRec's RQ2 (each heuristic alone vs combined) but adapted: there a variant optimizes a different per-query fitness (Eq. 6 / 7) with fixed α = β = 0.5; here the GA learns weights over only one family's features, same fitness, tested on unseen repos. GA-GRAPH has no counterpart (GA-IaCRec has no reference graph). Each fold uses the configuration its grid search chose (`grid.csv`), no re-tuning; 3 seeds per fold (seed SD 0.0008); the full GA reuses seeds 0–2 of the main run (identical settings hash). Reported: metrics of every variant, paired tests vs full (Wilcoxon + Holm, A12) and a **relative improvement** column RI = (mean_full − mean_variant) / mean_variant (GA-IaCRec's "RI"; descriptive only). Caveats for the report: `graph_propagation_conf` is mined from history but sits in the graph family by its prefix; under GA-GRAPH add-type candidates have all-zero features (only the offset places them), so the existing-block column is the fair one for that variant.
- **Learning curve:** subsample **whole training repos** (the unit the weights must generalize over; query subsampling is already answered by the fitness-sample check, `THRESHOLDS.md`). Fractions 10 / 25 / 50 / 75 % of each fold's training repos (≈ 4 / 10 / 21 / 32 repos) + 100 % = main run (seeds 0–2). 3 draws per fold and fraction; draw r uses a fixed permutation of the fold's training repos and GA seed r, so subsets are nested (10 % ⊂ 25 % ⊂ …). Same fixed per-fold configuration; test on the fold's core test repos. Reported: test MRR (both targets) and training-fitness MRR per fraction.
- **Runner:** both experiments run through the cross-validation runner (same folds, metrics, held-out add-only repos) into `data/results/ablation/` and `data/results/learning_curve/`; `tfimpact experiments --ablation` / `--learning-curve`; `tfimpact export --name ablation|learning_curve`. Settings in `experiments.*` (config).
- **Live test** (required by the course): procedure in `docs/LIVE_TEST.md`; prepared edits on repos outside the dataset, expected answers written before running, 2–3 raters, recorded under `results/live/<repo>/`.
- **External set** (optional, last step; **planned, not implemented** — code is added only if the step is taken): the same replay protocol as the evaluation (every qualifying commit, metrics of §9.1) on new repos, scoring only the final model. Selected exactly like the training repos (same filters, fork check also against the training repos), TerraDS star ranks **151–200**, no extension. Planned design: `--set external` on `select-repos` / `mine` (default `training` = current paths) writes `data/raw/external/{shortlist,repos,dataset}.csv`; mined tables share `data/processed/<slug>/` (membership is decided only by the dataset CSVs); a command scoring only the final model writes `data/results/external/`. Value: repos untouched by any design decision, and the shipped model itself; it is not new *kind* of evidence (same protocol as CV).
