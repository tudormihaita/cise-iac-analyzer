# Handoff — project status and how to resume

Last updated: 2026-11-18, after the report draft (milestone 7 in progress). Read this first when resuming; `docs/SPEC.md` stays
the source of truth for the design (decisions in §15), `docs/RESULTS.md` holds every result,
`docs/THRESHOLDS.md` the evidence for every setting.

## 1. The project in one paragraph

`tfimpact` (course project 2, Computational Intelligence in SE) recommends, for a Terraform block
a developer just edited (the *seed*), the other blocks that will likely change in the same commit
and the block types likely to be added. Data: 78 public AWS Terraform repos from TerraDS, mined
commit by commit (53 core repos + 8 add-only repos held out). Features: 12 per (seed, candidate)
in three families — association rules mined from history (block / type / file rules, propagation
rules over reference edges), the reference graph (proximity, Adamic–Adar shared neighbours,
degree), text (name and TF-IDF content similarity). Model: a linear score with an add-type
offset whose 13 parameters are learned by a self-implemented genetic algorithm (numpy only),
tuned per fold by nested grid search. Evaluation: grouped 5-fold cross-validation by repository.

## 2. Status

| Milestone | Content | Status |
|---|---|---|
| 1–2 | repo selection, mining pipeline, dataset (`select-repos`, `mine`, `build-dataset`) | done |
| 3 | features v2 (12 + 2 stored locality), causality tests | done |
| 4 | evaluation harness, baselines `arm` (ROSE), `fixed_weights`, `random`; stats | done |
| 5 | GA + random search, convergence check, nested grid search, sanity checks, final model | done |
| 6 | `recommend` tool, ablation, learning curve | done |
| **7** | report (`docs/report/`, drafted), README (done); live test optional, not in the report; external set optional | **in progress** |

Code state: 162 tests pass, `ruff check .` and `ruff format --check .` clean. Everything up to
the report draft is committed; the latest report edits and README are not (at the time of writing).

## 3. Key results (details: `docs/RESULTS.md` §5, files: `results/main/`)

Core repos, mean over repos, MRR (all targets / existing blocks only):

| Method | All targets | Existing blocks |
|---|---|---|
| **GA (tuned)** | **0.746** [0.727, 0.766] | **0.649** |
| random search (same budget) | 0.733 | 0.644 |
| `fixed_weights` (equal weights) | 0.563 | 0.644 |
| `arm` (ROSE) | 0.481 | 0.360 |
| `random` | 0.118 | 0.116 |

- GA beats ROSE on 53/53 repos and equal weights on 51/53 (Wilcoxon + Holm p < 1e-8, large A12);
  Hit@1 0.665, Hit@5 0.846; held-out add-only repos 0.716 (equal weights 0.177).
- Gain comes from add-type placement (new seeds 0.732 vs 0.388) and removed seeds (0.853 vs
  0.774, propagation rule). **On existing blocks GA = equal weights (0.649 vs 0.644, p = 0.19):**
  a linear score is saturated there → interactions / non-linear models are the natural next step.
- GA vs random search: consistent (50/53 repos, p < 1e-8) but small (+0.013, negligible A12):
  the 13-parameter problem is easy. Seed SD 0.0008, fold SD 0.03, repo SD 0.076.
- Sanity checks: shuffled labels → 0.428 (learning is real); negative weights allowed → same MRR,
  no negative mean weight (non-negativity justified).
- Learned weights: propagation 0.31, type rule 0.20, content similarity 0.18, offset 0.21.
- Grid search: all 27 configurations within 0.004–0.010; low mutation preferred. Final model
  (`models/ga_weights.json`): population 30, 120 generations, mutation 0.1, crossover 0.8.

## 4. Where things are

| Path | What |
|---|---|
| `src/tfimpact/collection/` | TerraDS selection, blobless clones |
| `src/tfimpact/processing/` | git walk, HCL parsing, references, snapshot, diff, filters, `pipeline.py` (single chronological pass = leakage boundary) |
| `src/tfimpact/features/` | `association_rules.py` (block/type/file), `propagation.py`, `graph.py`, `similarity.py`, `candidates.py`, `vector.py` (feature list `FEATURES`) |
| `src/tfimpact/evaluation/` | `loader.py`, `splits.py`, `metrics.py`, `stats.py`, `experiments.py` (CV runner), `convergence.py`, `figures.py`, `export.py` (copy to `results/`) |
| `src/tfimpact/methods/` | `base.py` (Method / Ranking), `baselines.py`, `ga/` package: `data.py` (training queries, sampling), `fitness.py`, `operators.py`, `search.py` (GA loop, random search), `grid.py` (nested grid search), `runs.py` (seeded streams, resumable run files), `method.py` (`GAMethod` + variants) |
| `src/tfimpact/inference/recommend.py` | `recommend`: mines the local repo (`process_repo` returns the final `RepoState`), seeds from `--block` or the working-tree diff, scores with the model, two lists; `tests/test_recommend.py` proves its features equal the training pipeline's |
| `models/ga_weights.json` | final model (committed) |
| `results/main/` | committed tables + figures of the main comparison |
| `data/` (gitignored) | `processed/` per-repo tables, `results/` full outputs + `ga_runs/`, `logs/` |

## 5. How to reproduce (from a fresh clone)

```bash
uv sync --extra plots
# TerraDS sqlite at data/external/terrads/TerraDS.sqlite (only needed for select-repos)
uv run tfimpact mine                 # ~14 min, pinned commits in data/raw/repos.csv, builds dataset.csv
uv run tfimpact evaluate             # ~3 h: GA (grid search per fold) + baselines; resumable
uv run tfimpact evaluate --methods ga_permuted,ga_negative,fixed_weights --out checks   # ~5 min
uv run tfimpact experiments --convergence   # ~20 min
uv run tfimpact train                # final model -> models/ga_weights.json
uv run tfimpact export               # copy tables + figures -> results/main/
uv run tfimpact experiments --ablation --learning-curve   # ~40 min, reuses grid.csv + main runs
uv run tfimpact export --name ablation && uv run tfimpact export --name learning_curve
```

Long runs are resumable (per-run JSON in `data/results/ga_runs/`, settings-hashed); start them in
the background and re-run the same command after an interruption.

## 6. Next steps

**Milestone 6 (done):**
- `tfimpact recommend --repo PATH (--block ADDRESS | --diff) [--top 10] [--explain] [--model PATH]
  [--json]` (SPEC §10, §15.16). Smoke test on `terraform-aws-cloudfront`, `docs/RESULTS.md` §6.
- Ablation (`tfimpact experiments --ablation`, `results/ablation/`, RESULTS §7): GA-HIST 0.660,
  GA-TEXT 0.488, GA-GRAPH 0.420 vs full 0.745; full better on 53/53 repos. Families are
  complementary: history → added seeds, graph (propagation) → removed seeds, text → modified
  seeds / existing blocks.
- Learning curve (`tfimpact experiments --learning-curve`, `results/learning_curve/`, RESULTS
  §8): flat from ~10 training repos (10 %: −0.011; 25–75 %: n.s.); no train/test gap → the
  linear model, not data, limits quality.
- Both run through the CV runner with each fold's grid-chosen configuration (`grid.csv`), 3
  seeds; the full GA reuses the main run's files. Settings in `experiments.*`.

**Milestone 7 (in progress):**
1. **Report** — LaTeX in `docs/report/` (`main.tex`, `chapters/01–09`, `biblio.bib`, `fig/`),
   all sections drafted; target 8–10 pages. Decisions with the author: RQ1 = is the GA a
   feasible and reliable learner (vs random ranking, shuffled labels, convergence, seeds,
   random search); RQ2 = learned ranking vs ROSE and equal weights; the ablation is a results
   subsection without its own RQ, the learning curve one closing paragraph (never a headline);
   **no live-test / user-feedback section**; §3 "Problem Statement"; Discussion short and feeding
   future work. Tables report mean ± SD over repos and 95 % CI. Not compiled here (no LaTeX):
   if it is over 10 pages, trim in this order: learned-weights figure, positioning table rows,
   merge §6 statistics into comparisons, funnel table into text. `shaw2003writing` is in the bib
   but not cited; TerraDS page numbers in MSR 2025 still to check.
2. **README** — commands for every step incl. experiments, results table: done.
3. **Live test** (optional, not part of the report): `docs/LIVE_TEST.md`.
4. **External set** (optional, SPEC §15.17, **not implemented**).
5. `tfimpact report` (SPEC §10) is not implemented; the LaTeX report replaces it.

**Optional extensions discussed** (only after 6–7): context-dependent GA (weights per seed
action), LightGBM LambdaMART as a non-linear reference, relational association rules as a second
learner, Apriori over enriched candidate descriptions, GP (SPEC stretch).

## 7. Working agreements with the author

- The author **commits**; propose a commit message, never commit or push.
- **Explain and propose before deciding**: thresholds, feature/metric/protocol changes and design
  choices are confirmed by the author first (CLAUDE.md invariant 7); every threshold needs
  evidence (`docs/THRESHOLDS.md`). Data is excluded only with measurable justification.
- The author asks many conceptual questions: answer them clearly before acting; use concrete
  examples and measured numbers.
- Code comments describe behaviour only: no `SPEC §` references or changelogs in code.
- Throwaway prototypes and analyses go to a scratchpad / temp folder, not `src/` or `data/`.
- Record every milestone's results in `docs/RESULTS.md`; keep SPEC §15 decisions current.
- Long runs: background + resumable; report progress from the saved run files.

## 8. Gotchas

- `data/` is gitignored: a new machine needs the re-mine (~14 min) and evaluation (~3 h), or a
  copy of `data/processed/` and `data/results/`. `recommend` itself only needs `models/`.
- `PIPELINE_VERSION` (`constants.py`) must be bumped when code changes alter per-repo tables;
  `settings_hash` covers config changes. GA runs are keyed by a settings hash too.
- `docs/` also contains the paper PDFs and `requirements.pdf`: keep them out of a public repo.
- `ruff format` re-adds blank lines after class docstrings; macOS has no `timeout` command.
- Leakage checks exist and must stay green: `tests/test_rules_causality.py` (prefix invariance,
  independent recomputation of rules / content similarity).

Suggested commit messages for the uncommitted work (two commits, or one combined):

```
Add `tfimpact recommend` inference tool (milestone 6, part 1)

Mines a local repo's history with the training pipeline, takes seeds from a
block address or the working-tree diff, scores candidates with the trained
model (models/ga_weights.json) and prints blocks to modify and block types to
add (--explain, --json). Tests prove the features equal the training
pipeline's for the same edits; smoke test on an unseen repo in RESULTS.md §6.
```

```
Add ablation study and learning curve (milestone 6, part 2)

`tfimpact experiments --ablation` trains one GA per feature family (hist,
graph, text) and compares each with the full GA; `--learning-curve` trains
on 10-75 % of each fold's training repos. Both reuse the folds and each
fold's grid-chosen configuration; the full GA loads the main run's saved
runs. Paired statistics gain a relative-improvement column. Results in
results/ablation/, results/learning_curve/ and RESULTS.md §7-8; live-test
guide in docs/LIVE_TEST.md; decisions in SPEC §15.17. Also fixes a crash at
the end of `select-repos` (output path shadowed the print helper).
```

## 9. Prompt to resume in a new Claude Code session

```
I'm resuming work on the tfimpact project (Terraform change-impact recommender, course
project). Read CLAUDE.md, then docs/HANDOFF.md (status, results, next steps, working
agreements), then the relevant parts of docs/SPEC.md (§15 holds all decisions) and
docs/RESULTS.md. Milestones 1–6 are done and the report in docs/report/ is drafted (HANDOFF §6).
In this session we finish the report (compile, page count, final edits). Before implementing, summarize your understanding
of the current state and propose the exact plan for each part (variants, seeds, run time,
outputs, how the live checks are recorded) so I can confirm it. Follow the working agreements in
HANDOFF §7 (I commit myself; propose decisions before making them; no SPEC references in code
comments).
```
