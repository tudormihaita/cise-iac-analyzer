# Results log

Results obtained so far, in the order they were produced. Raw outputs live in `data/results/`
(gitignored); this file is the committed record. Decisions behind each step: `docs/SPEC.md` §15,
threshold evidence: `docs/THRESHOLDS.md`.

Regenerate: `uv run tfimpact mine` (pinned commits, `PIPELINE_VERSION` 3), then
`uv run tfimpact evaluate`.

## 1. Dataset (feature set v2 re-mine, 2026-11-16)

| | Value |
|---|---|
| Repos mined (pinned `head_sha` in `data/raw/repos.csv`) | 78 |
| **Core repos** (included by `build-dataset`) | **53** (26 deployment, 21 module library, 6 educational) |
| Add-only repos (excluded only for `add_only_frac > 0.4`), held out in evaluation | 8 |
| Excluded: < 50 usable commits / both reasons | 14 / 3 |
| Non-warm-up queries, core / add-only | 55,541 / 6,719 |
| Feature rows (core) | 11.9 M (≤ 200 candidates per query) |
| Candidate recall ceiling (mean over repos), core / add-only / all | 0.810 / 0.671 / 0.792 |
| Mining time, all 78 repos | 13.9 min |

## 2. Feature analyses (milestone 3)

### 2.1 Graph distance vs chance of being a true answer (feature set v1, 53 repos)
Existing candidates, non-warm-up queries; mean over repos. Motivated dropping `ref_direct`
(identical to `graph_proximity = 0.5` on every row).

| Distance | P(true answer) | Share of all true answers |
|---|---|---|
| 1 | 10.8 % | 7.9 % |
| 2 | 11.8 % | 28.1 % |
| 3 | 4.5 % | 12.7 % |
| 4 | 3.8 % | 8.1 % |
| > 4 / unreachable | 2.3 % | 43.2 % |

Direct neighbours of a **removed** seed are true answers 58.5 % of the time (modified 6.5 %,
added 9.0 %), which motivated `graph_propagation_conf`.

### 2.2 Rule coverage (feature set v1)
Share of true existing answers with any earlier co-change evidence: block level 27 %; 72 % of
answers lie in another file than the seed, and 30 % of those have file-level but no block-level
history. 39 % of seeds are new blocks. Motivated `hist_file_cochange_conf`.

### 2.3 Locality ablation (proxy, feature set v1)
Random weight search (800 tries), grouped 5-fold by repo, ≤ 400 queries per repo, MRR:

| Search seed | All 12 v1 features | Without `same_file` + `same_dir` | Difference |
|---|---|---|---|
| 1 | 0.624 | 0.635 | +0.011 |
| 2 | 0.610 | 0.625 | +0.016 |
| 3 | 0.637 | 0.627 | −0.010 |

Differences are smaller than the spread between seeds (0.027): locality is stored but not
used by default. Same-directory blocks unreachable in the graph are still 3.3× more likely to
be answers than other unreachable blocks (univariate information, absorbed by other features
in combination).

### 2.4 Per-feature strength, feature set v2 (each feature alone as a ranking, 53 core repos)

| Feature | MRR all | MRR existing | MRR add-type | non-zero true / false |
|---|---|---|---|---|
| `text_content_similarity` | 0.465 | **0.530** | – | 0.90 / 0.79 |
| `hist_type_cochange_conf` | 0.429 | 0.254 | **0.670** | 0.71 / 0.59 |
| `text_name_similarity` | 0.348 | 0.389 | – | 0.43 / 0.23 |
| `graph_shared_neighbors` (Adamic–Adar; Jaccard was 0.376) | 0.342 | 0.384 | – | 0.34 / 0.11 |
| `hist_cochange_conf` / `_count` / `_recency` | 0.341–0.343 | 0.372–0.375 | – | 0.30 / 0.04 |
| `hist_cand_recent_changes` | 0.296 | 0.328 | – | 0.60 / 0.35 |
| `graph_proximity` | 0.255 | 0.289 | – | 0.61 / 0.39 |
| `hist_file_cochange_conf` | 0.255 | 0.285 | – | 0.63 / 0.32 |
| `graph_propagation_conf` | 0.253 | 0.284 | – | 0.09 / 0.03 |
| `graph_cand_degree` | 0.228 | 0.254 | – | 0.97 / 0.98 |
| `loc_same_file` (stored only) | 0.278 | 0.319 | – | 0.31 / 0.08 |
| `loc_same_dir` (stored only) | 0.197 | 0.227 | – | 0.61 / 0.30 |
| random | 0.118 | 0.119 | 0.324 | |

"–": the feature is 0 for add-type candidates (ranking falls back to random, ≈ 0.324).
Only rank correlations ≥ 0.6: the three block-history features (≈ 1.0 on all rows, 0.90–0.93
where history exists). `loc_same_file` vs `hist_file_cochange_conf`: 0.35.

**Propagation rule rediscovers Terraform's dependency rule from data** (direct neighbours,
mean over repos):

| Seed action | Learned value | Actual answer rate |
|---|---|---|
| removed | 0.564 | 0.585 |
| modified | 0.063 | 0.065 |
| added | 0.094 | 0.090 |

## 3. Baselines (milestone 4, 2026-11-17)

Grouped 5-fold CV over the 53 core repos (`data/results/folds.csv`); the 8 add-only repos are
never trained on and evaluated as a held-out group. Mean over repos, MRR ± SD over repos
[bootstrap 95 % CI].

### 3.1 Main table

| Method | Subset | MRR, all targets | MRR, existing blocks |
|---|---|---|---|
| `arm` (ROSE) | core | 0.481 ± 0.152 [0.443, 0.522] | 0.360 ± 0.164 [0.318, 0.405] |
| | add-only | 0.667 ± 0.162 [0.559, 0.759] | 0.268 ± 0.167 [0.168, 0.382] |
| | all 61 | 0.506 ± 0.164 [0.467, 0.545] | 0.348 ± 0.166 [0.311, 0.390] |
| `fixed_weights` | core | **0.563** ± 0.147 [0.526, 0.601] | **0.644** ± 0.106 [0.617, 0.673] |
| | add-only | 0.177 ± 0.118 [0.104, 0.250] | 0.630 ± 0.129 [0.553, 0.719] |
| | all 61 | 0.513 ± 0.194 [0.467, 0.562] | 0.642 ± 0.108 [0.614, 0.671] |
| `random` | core | 0.118 ± 0.037 | 0.116 ± 0.037 |
| | add-only | 0.102 ± 0.041 | 0.135 ± 0.090 |

Other core metrics, all targets (`arm` / `fixed_weights` / `random`): MAP 0.231 / 0.332 / 0.053,
Hit@1 0.426 / 0.473 / 0.038, Hit@5 0.555 / 0.670 / 0.162, Recall@10 0.283 / 0.344 / 0.043,
Precision@5 0.267 / 0.337 / 0.037. Existing blocks: MAP 0.211 / 0.444 / 0.058, Hit@5 0.440 /
0.769 / 0.160, Recall@10 0.278 / 0.490 / 0.050. SD over folds (MRR, all): 0.039 / 0.063 / 0.016.

### 3.2 Paired comparisons (core repos, Wilcoxon signed-rank + Holm, A12)

| Target, metric | `arm` vs `fixed_weights`: repos better | p (Holm) | A12 (`arm`) |
|---|---|---|---|
| all, MRR | 14 / 39 | 0.0015 | 0.34 (medium) |
| all, MAP | 7 / 46 | < 0.0001 | 0.24 (large) |
| existing, MRR | 0 / 53 | < 0.0001 | 0.09 (large) |
| existing, MAP | 0 / 53 | < 0.0001 | 0.09 (large) |

Both methods beat `random` on (almost) every repo (p < 0.0001, A12 ≥ 0.94).

### 3.3 Breakdowns (core, MRR)

| Group | `arm` all | `fixed_weights` all | `arm` existing | `fixed_weights` existing |
|---|---|---|---|---|
| seed added | **0.507** | 0.388 | 0.154 | **0.505** |
| seed modified | 0.414 | **0.640** | 0.430 | **0.665** |
| seed removed | 0.335 | **0.774** | 0.343 | **0.788** |
| deployment | 0.485 | 0.562 | 0.360 | 0.659 |
| module library | 0.478 | 0.566 | 0.365 | 0.628 |
| educational | 0.477 | 0.559 | 0.344 | 0.635 |

File level (supplementary): MRR `arm` 0.479, `fixed_weights` 0.768, `random` 0.338.

### 3.4 Observations
- Combining all 12 signals with equal weights already beats ROSE on 39 of 53 repos (all
  targets) and on every repo for existing blocks: the additional signals pay off before any
  learning.
- Equal weights are weak exactly where answers are **block types to add**: new seeds (0.388 vs
  `arm` 0.507) and the add-only repos (0.177 vs 0.667), where most answers are add-type and
  only two features are non-zero for add-type candidates. The type rule alone (`arm`'s fallback)
  ranks them far better. This is the main thing learned weights should fix.
- No difference between repo kinds.
- File level is much easier (random 0.338), so it stays a side remark.

## 4. GA fitness-sample check (milestone 5 preparation, 2026-11-17)

Throwaway prototype of the agreed GA (12 non-negative weights + add-type offset, MRR fitness on
all targets, population 60 × 60 generations, SPEC §8.1 operators), trained on fold 0's 43 core
training repos (43,225 queries with a true answer among the candidates, 9.3 M rows) and tested
on its 10 core test repos. 3 seeds per sample size; sample = equal number of queries per repo.

| Fitness sample | Time per run | Training MRR (all training queries) | Test MRR (mean ± SD over seeds) |
|---|---|---|---|
| 2,000 queries | 7 s | 0.750 ± 0.003 | 0.701 ± 0.008 |
| 5,000 queries | 20 s | 0.753 ± 0.000 | **0.705 ± 0.000** |
| all queries | 182 s | 0.754 ± 0.001 | 0.706 ± 0.003 |

Same fold-0 test repos for reference: `fixed_weights` 0.534, `arm` 0.437, `random` 0.123.

- 5,000 queries reach the all-data test MRR (difference 0.001) at 1/9 of the cost; 2,000 is
  0.005 lower and noisier across seeds.
- Convergence: the best fitness reaches 99 % of its final value by generation 11–32, so
  60 generations is enough at population 60 (to be confirmed by the convergence check).
- Learned weights are consistent across sample sizes and seeds (mean, 5,000 sample):
  `graph_propagation_conf` 0.37, `hist_type_cochange_conf` 0.19, `text_content_similarity` 0.16,
  `hist_cochange_recency` 0.08, `graph_shared_neighbors` 0.07, all others ≤ 0.04; add-type
  offset b ≈ 0.18–0.23 (positive: add-type candidates are lifted up the list).

### 4.1 Chromosome variant and label-permutation check (fold 0, 5,000 sample, 3 seeds)

| Variant | Test MRR all targets | Test MRR existing blocks | Add-type offset b |
|---|---|---|---|
| **13 genes** (12 shared weights + offset b) | 0.705 ± 0.000 | 0.609 ± 0.003 | 0.19 |
| 15 genes (+ own add-type weights for type rule and name) | 0.707 ± 0.003 | 0.612 ± 0.004 | 0.04 |
| 13 genes, **labels shuffled** within each training query | 0.440 ± 0.047 | 0.505 ± 0.057 | −0.74 |
| (`fixed_weights`, same test repos) | 0.534 | 0.613 | – |

- Separate add-type weights change *how* the add-type lift is learned (the slope v_type ≈ 0.5
  replaces most of the offset, and the shared type-rule weight drops from 0.19 to 0.12) but not
  the result: +0.002 / +0.003, within the seed-to-seed spread. **The 13-gene design is kept.**
- **Label permutation:** with shuffled training labels the training fitness falls to chance
  (0.12) and test MRR drops from 0.705 to 0.440, below `fixed_weights`. The gain comes from
  learning real labels, not from information leaking into training. (Test MRR stays above
  `random` because any non-negative combination of the causal features is informative.)
- On fold 0, the GA's gain over `fixed_weights` is entirely on all targets (add-type placement);
  on existing blocks both are ≈ 0.61. To be confirmed on all folds.

## 5. GA with nested grid search: main comparison (milestone 5, 2026-11-17)

`tfimpact evaluate` (~3 h): per fold, the grid search (27 configurations × 3 seeds, trained on
75 % of the fold's core training repos, scored on the other 25 %) picks a configuration, which
is trained with 10 seeds on all training repos and tested on the fold's core test repos and the
8 held-out add-only repos. `random_search` uses the same evaluation budget (10 seeds per fold).
Report files and figures: `results/main/` (copied by `tfimpact export`); final model:
`models/ga_weights.json` (`tfimpact train`).

### 5.1 Main results (core repos, mean over repos, MRR ± SD over repos [95 % CI])

| Method | MRR all targets | MRR existing blocks | MAP all | Hit@1 all | Hit@5 all | Recall@10 all |
|---|---|---|---|---|---|---|
| **`ga`** | **0.746** ± 0.076 [0.727, 0.766] | **0.649** ± 0.110 [0.622, 0.680] | **0.437** | **0.665** | **0.846** | **0.455** |
| `random_search` | 0.733 ± 0.079 [0.713, 0.754] | 0.644 ± 0.110 | 0.430 | 0.650 | 0.835 | 0.448 |
| `fixed_weights` | 0.563 ± 0.147 [0.526, 0.601] | 0.644 ± 0.106 | 0.332 | 0.473 | 0.670 | 0.344 |
| `arm` (ROSE) | 0.481 ± 0.152 [0.443, 0.522] | 0.360 ± 0.164 | 0.231 | 0.426 | 0.555 | 0.283 |
| `random` | 0.118 ± 0.037 | 0.116 ± 0.037 | 0.053 | 0.038 | 0.162 | 0.043 |

Held-out add-only repos (never trained on), MRR all / existing: `ga` 0.716 / 0.671,
`random_search` 0.713 / 0.655, `arm` 0.667 / 0.268, `fixed_weights` 0.177 / 0.630.
All 61 repos: `ga` 0.742 / 0.652. Candidate recall ceiling: 0.810 (core).

Spread of `ga` (MRR, all targets): SD over repos 0.076, over folds 0.032, over GA seeds 0.0008
(`random_search`: 0.0031). File level: `ga` 0.774, `fixed_weights` 0.768, `random_search` 0.769.

### 5.2 Paired comparisons (core repos, Wilcoxon signed-rank + Holm, A12)

| `ga` vs | Target, metric | Repos better / worse | p (Holm) | A12 |
|---|---|---|---|---|
| `arm` | all, MRR | 53 / 0 | < 1e-8 | 0.93 (large) |
| `fixed_weights` | all, MRR | 51 / 2 | < 1e-8 | 0.87 (large) |
| `random_search` | all, MRR | 50 / 3 | < 1e-8 | 0.56 (negligible) |
| `arm` | existing, MRR | 53 / 0 | < 1e-8 | 0.91 (large) |
| `fixed_weights` | existing, MRR | 32 / 21 | 0.19 | 0.52 (negligible) |
| `random_search` | existing, MRR | 41 / 12 | 0.0005 | 0.53 (negligible) |
| `fixed_weights` | existing, MAP | 26 / 27 | 0.54 | 0.47 (negligible) |

### 5.3 Breakdowns (core, MRR)

| Seed action | `ga` all | `fixed_weights` all | `arm` all | `ga` existing | `fixed_weights` existing |
|---|---|---|---|---|---|
| added | **0.732** | 0.388 | 0.507 | 0.461 | **0.505** |
| modified | **0.690** | 0.640 | 0.414 | **0.685** | 0.665 |
| removed | **0.853** | 0.774 | 0.335 | **0.869** | 0.788 |

By repo kind (all targets): deployment 0.754, educational 0.754, module library 0.733.

### 5.4 Sanity checks (`results/checks/`, 1 run per fold)

| Variant | MRR all | MRR existing | Reading |
|---|---|---|---|
| `ga_permuted` (labels shuffled in training) | 0.428 | 0.461 | collapses below `fixed_weights`: the gain comes from learning real labels |
| `ga_negative` (weights may be negative) | 0.742 | 0.641 | same as `ga`; the most negative weight per run is −0.002 … −0.069, mean over runs ≥ −0.001 for every feature: no suppressor, non-negative weights justified |

### 5.5 Grid search and learned model

- Validation MRR spread across the 27 configurations: 0.004–0.010 per fold; chosen vs default
  configuration (60 × 60, mutation 0.3, crossover 0.8): +0.001 to +0.008. The GA is insensitive
  to its settings; the only consistent preference is a **low mutation rate (0.1)** (4 of 5 folds).
- Chosen per fold: `p30g120_m0.1_c0.8` (folds 0, 2), `p60g60_m0.1_c0.8` (1),
  `p30g120_m0.3_c0.95` (3), `p30g120_m0.1_c0.95` (4). Final model (`models/ga_weights.json`, all 53
  core repos): `p30g120_m0.1_c0.8`.
- Learned weights (`ga`, mean ± SD over 50 runs): `graph_propagation_conf` 0.31 ± 0.05,
  `hist_type_cochange_conf` 0.20 ± 0.03, `text_content_similarity` 0.18 ± 0.03,
  `graph_shared_neighbors` 0.06, `hist_cochange_recency` 0.05, `graph_proximity` 0.05, all
  others ≤ 0.04; add-type offset 0.21 ± 0.02.

### 5.6 Observations
- Learning the weights pays off: `ga` beats ROSE (+0.27 MRR) and equal weights (+0.18) on
  almost every repo, with large effect sizes, and generalizes to the held-out add-only repos
  (0.716 vs 0.177 for equal weights). It is also far more consistent across repos (SD 0.076 vs
  0.147).
- **Where the gain comes from:** placing add-type candidates (new seeds: 0.732 vs 0.388) and
  removed seeds (0.853 vs 0.774, driven by the propagation rule). On existing blocks alone `ga`
  equals `fixed_weights` (0.649 vs 0.644, not significant): equal weights are already close to
  the best *linear* combination there, which points to interactions/non-linear models as the
  next step rather than more weight tuning.
- **GA vs random search:** the GA wins consistently (50 of 53 repos, p < 1e-8) and is 4× more
  stable across seeds, but the margin is small (+0.013, negligible A12). With 13 continuous
  parameters and 3,660 evaluations, random search already gets close: the search problem is
  easy. The GA is a sound optimizer here, not a decisive one.

## 6. `recommend` smoke test on an unseen repo (milestone 6, 2026-11-17)

`terraform-aws-modules/terraform-aws-cloudfront` (not among the 78 mined repos; 138 commits),
final model `models/ga_weights.json`; 2.7 s per call including mining the history.

- `--block aws_cloudfront_function.this`: 1. `aws_cloudfront_distribution.this` (attaches the
  function), 2. the example's `module.cloudfront` call, 3. `wrappers::module.wrapper`,
  4. `var.cloudfront_functions` (the function's input).
- `--diff` after renaming only the variable `cloudfront_functions` → `functions_config` in
  `variables.tf` (uncommitted): seeds = removed old + added new variable; top 3 =
  `examples/complete::module.cloudfront` and `wrappers::module.wrapper` (both still pass the
  old argument, linked through module plumbing) and `aws_cloudfront_function.this` (reads the
  old variable), all driven by `graph_propagation_conf`: exactly the blocks the rename breaks.
  Types to add: variable, output, local, module.

## 7. Ablation: one GA per feature family (milestone 6, 2026-11-18)

`tfimpact experiments --ablation` → `results/ablation/`. Per fold the configuration the main
run's grid search chose (`grid.csv`), 3 seeds; the full GA reuses seeds 0–2 of the main run
(0.745 vs 0.746 with 10 seeds). Core repos, mean over repos; paired tests full vs variant on
per-repo MRR (Wilcoxon + Holm, A12); RI = (full − variant) / variant (GA-IaCRec's measure).

| Variant | Features | MRR all [95 % CI] | MRR existing | Hit@5 all | Full better / worse (all) | RI full (all / existing) | A12 (all / existing) |
|---|---|---|---|---|---|---|---|
| **GA (full)** | 12 | **0.745** [0.727, 0.766] | **0.648** | **0.846** | — | — | — |
| GA-HIST | 6 | 0.660 [0.635, 0.685] | 0.497 | 0.762 | 53 / 0 | +12.9 % / +30.5 % | 0.76 / 0.81 |
| GA-TEXT | 2 | 0.488 [0.445, 0.533] | 0.533 | 0.576 | 53 / 0 | +52.9 % / +21.6 % | 0.92 / 0.76 |
| GA-GRAPH | 4 | 0.420 [0.392, 0.450] | 0.460 | 0.543 | 53 / 0 | +77.7 % / +40.9 % | 1.00 / 0.90 |

All differences p (Holm) < 1e-8. By seed action (MRR all targets: added / modified / removed):
full 0.733 / 0.690 / 0.848, HIST 0.715 / 0.584 / 0.548, TEXT 0.273 / 0.615 / 0.582, GRAPH
0.351 / 0.415 / 0.778.

Observations:
- **Every family alone is clearly worse than the combination** (full GA better on 53/53 repos
  for all targets, 50–52/53 for existing blocks, large effects): the families are
  complementary, as GA-IaCRec found for history + similarity (RQ2), here with learned weights
  and a third family.
- **Each family covers a different case:** history carries **added** seeds (0.715 of the full
  0.733: the type rule places add-type candidates; HIST weight 0.46 on
  `hist_type_cochange_conf`, offset 0.42); the graph carries **removed** seeds (0.778 of 0.848:
  `graph_propagation_conf`, weight 0.52); text is the best single family on **modified** seeds
  and on existing blocks overall (0.533; `text_content_similarity` weight 0.91).
- Ranking of single families depends on the target set: all targets HIST > TEXT > GRAPH
  (add-type placement needs the type rule); existing blocks TEXT > HIST > GRAPH. GA-GRAPH's
  all-target number is pessimistic by construction (add-type candidates have all-zero graph
  features, only the offset places them), so its existing-block column is the fair one.
- Caveat: `graph_propagation_conf` is mined from history but counted in the graph family by
  its name; GA-GRAPH's strength on removed seeds is partly a history signal.

## 8. Learning curve: training repos per fold (milestone 6, 2026-11-18)

`tfimpact experiments --learning-curve` → `results/learning_curve/` (figure
`figures/learning_curve.png`). GA trained on 10 / 25 / 50 / 75 % of each fold's core training
repos (3 nested draws per share; each fold's grid-chosen configuration), tested on the fold's
core test repos; 100 % = main run (seeds 0–2). Training MRR = fitness on the run's training
sample (all targets).

| Share | Training repos | Test MRR all [95 % CI] | Test MRR existing | Training MRR | Full better / worse (all) | p (Holm) | A12 |
|---|---|---|---|---|---|---|---|
| 10 % | 4.0 | 0.734 [0.714, 0.754] | 0.641 | 0.746 | 48 / 5 | < 1e-6 | 0.55 |
| 25 % | 10.4 | 0.743 [0.723, 0.764] | 0.648 | 0.751 | 34 / 19 | 0.055 | 0.51 |
| 50 % | 21.4 | 0.747 [0.728, 0.767] | 0.652 | 0.752 | 23 / 30 | 0.093 | 0.49 |
| 75 % | 32.0 | 0.744 [0.725, 0.765] | 0.648 | 0.748 | 32 / 21 | 0.055 | 0.51 |
| 100 % | 42.4 | 0.745 [0.727, 0.766] | 0.648 | 0.748 | — | — | — |

Held-out add-only repos (MRR all): 0.711 (10 %) … 0.716 (50–100 %). SD over draws ≤ 0.008,
SD over folds 0.03–0.05.

Observations:
- **The curve is flat from about 10 training repos:** 10 % (4 repos) loses 0.011 MRR (small
  but consistent, 48/53 repos, negligible A12); from 25 % on, no share differs significantly
  from the full training set. Learning happens (permuted labels: 0.428, §5.4; equal weights:
  0.563), but 13 parameters are learned from very few repos.
- **No overfitting gap:** training MRR (0.746–0.752) is within 0.01 of test MRR at every
  share, so the weights generalize across repos and more data cannot close a gap.
- Consequence: **the linear model, not the amount of data, limits quality.** More repos would
  not improve the GA; richer models (interactions, per-seed-action weights, LambdaMART, GP)
  are the way forward, consistent with GA = equal weights on existing blocks (§5.6). It also
  means the shipped model would be the same if trained on a few repos of a user's own
  organisation.
