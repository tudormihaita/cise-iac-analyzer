# Thresholds and filters — values and justifications

Every data-selection threshold in `configs/default.yaml`, with the reason for its value.
Numbers marked *run 1* come from the first mining run (40 repos, 2026-11-08). Confirmed by
the author on 2026-11-08. Changing a value here means changing the config, and vice versa.

## Repository selection (`selection.*`)

| Parameter | Value | Applied | Justification |
|---|---|---|---|
| `provider` | `aws` | TerraDS query | AWS is the most common provider in TerraDS (33,369 repos manage AWS resources vs. 10,181 for `azurerm`). A single provider gives all repos one shared vocabulary of resource types, so type-level rules (`hist_type_cochange_conf`) learned on training repos transfer to unseen test repos under cross-repository CV. The code is provider-agnostic; only this value names a provider. |
| `min_resources` | 20 | TerraDS query | With fewer than ~20 managed resources the candidate list is so short that even random ranking scores high, and graph features (paths, neighbours) are nearly empty. Such repos would inflate metrics without testing the method. 9,619 TerraDS repos pass. |
| Archived repos | included | TerraDS query | An archived repo's history is frozen but real; it is still valid evidence of how Terraform code co-evolves (and cannot change between runs). 323 of 9,619 candidates are archived. |
| `shortlist` | 150 (top by stars) | TerraDS query | A **budget**, not a methodological threshold. Stars order candidates by popularity/maturity; 150 is about 2× the 25–30-repo target, given that ~45 % pass the depth check and ~80 % survive `build-dataset` (run 1), while keeping selection and mining within the project's time budget. `null` / `select-repos --all` depth-checks every TerraDS candidate (planned future run). |
| `min_tf_commits` | 50 | depth check (blobless, before mining) | Cheap proxy for `min_usable_commits`. A usable commit must touch a `.tf` file, so *usable ≤ `.tf` commits* within the same window. Setting the proxy equal to `min_usable_commits` makes it **lossless**: it only discards repos that certainly fail the exact filter. (Run 1 usable/`.tf` ratio was 0.40–0.75, so any higher proxy could wrongly drop repos.) |
| `dedup_forks` | true | after the depth check | Forks and copies share commit history. If two of them landed in different CV folds, the test repo's history would already have been seen in training (leakage across folds). Repos sharing a root commit count as one project; the most-starred one is kept. |
| `min_usable_commits` | 50 | `build-dataset` (after mining) | Each usable commit yields ~4–5 queries. After the 20 % warm-up, 50 usable commits leave ≥ 40 query commits (~150+ queries): enough for a stable per-repo MRR, for the per-repo paired tests (Wilcoxon), and for a non-empty 10 % temporal test split (protocol P2). |
| `max_add_only_frac` | 0.4 | `build-dataset` (after mining) | **Why exclude:** in an add-only commit the seed is a brand-new block, so the block-history features (`hist_cochange_*`, `hist_cand_recent_changes`) are 0, and every positive is an add-type candidate that only the type rule (`hist_type_cochange_conf`) can score. A repo made almost entirely of such commits (tutorial lesson folders, generated examples) measures "what is written together", not change impact. Because CV weights repos equally, each such repo pulls the learned weights toward the type rule. **Why 0.4:** run 1 was bimodal. 34 repos fall in [0.02, 0.32], then there is a gap, then 6 repos fall in [0.45, 0.96]: `terraform-provider-awscc` 0.96, `antonputra/tutorials` 0.93, `devopsacademyau/academy` 0.88, `terraform-tuesdays` 0.65, `config-lint` 0.53, `kubernetes-ops` 0.45. Any value in [0.33, 0.44] gives the same split; 0.4 is the round value inside the gap. **Confirmed on run 2** (78 repos, 38 of them new): among repos with ≥ 50 usable commits the gap persists, with the highest kept repo at 0.316 (`modernisation-platform`) and the lowest excluded one at 0.452 (`ManagedKube/kubernetes-ops`), so the threshold, fixed before run 2, generalised to unseen repos. **Caveat:** chosen after seeing the data, so these repos are never trained on but always evaluated as a held-out group, reported next to the core results (core 53 / add-only 8 / all 61; `docs/RESULTS.md` §3). Only whole repos are dropped; add-only commits inside kept repos remain. |

## Mining (`mining.*`)

| Parameter | Value | Justification |
|---|---|---|
| `max_commits` | 2000 (most recent, first-parent) | Bounds runtime and focuses on recent practice. In run 1 only 5 of 40 repos reached the cap; for the others it is their full history (final dataset: 5 of 53 core repos). `recommend` uses the same limit, so it mines history under the conditions the model was evaluated in. |
| `min_items` | 2 | Required by the task definition: labels are the commit's *other* changed blocks, so a single-item commit has no label and cannot form a query. (27.8 % of commits with changes are single-item, run 1.) |
| `max_items` | 30 | Large commits are bulk refactors, new modules or mass renames ("tangled" commits). Their co-changes are largely coincidental and pair counts grow quadratically. Run 1: median 4 changed blocks per commit, 90th percentile 36; the cut drops 11.8 % of commits with changes. Same cut-off as ROSE (Zimmermann et al., ICSE 2004, §3: "ROSE does so by ignoring all changes that affect more than 30 entities", introduced there to avoid large merge transactions). |
| `warmup_frac` | 0.2 | The first usable commits would be scored by an almost empty rule miner, understating history-based methods. The first 20 % feed the miner only (no queries). |
| `use_merge_commits` | true | Many deployment repos use merge-commit PRs, so their changes appear only in first-parent merge commits (`alphagov/govuk-aws`: 1 non-merge vs. 1,313 merge `.tf` commits). Excluding merges would leave mostly module libraries. A merge is diffed against its first parent, giving one PR-level transaction (same granularity as a squash merge); `max_items` still drops oversized PRs. `is_merge` is stored for a with/without sensitivity check. See SPEC §15.4. |

## Feature computation (`features.*`)

Measured on the 3 dev repos (`terraform-aws-vpc` module library, `poseidon/typhoon` linear deployment, `alphagov/govuk-aws` merge-heavy deployment; ~10k non-warm-up queries) with a widened budget (1000 candidates, graph distance 6). The dev repos remain in the evaluation; this is disclosed as a threat to validity.

| Parameter | Value | Evidence / justification |
|---|---|---|
| `max_candidates` | 200 | Share of true existing-block targets kept in the candidate list (vpc / typhoon / govuk): @100 = 75 / 80 / 87 %, **@200 = 81 / 90 / 91 %**, @400 = 89 / 95 / 93 %, @1000 = 95 / 100 / 96 %. 200 is the knee: doubling to 400 adds only 2–8 points for twice the data and runtime. The resulting ceiling is reported (no method can exceed it). |
| `graph_max_dist` | 4 | Share of true targets within d reference hops of the seed (vpc / typhoon / govuk): ≤ 3 = 70 / 10 / 58 %, **≤ 4 = 75 / 12 / 70 %**, ≤ 6 = 76 / 12 / 77 %. Beyond 4 hops almost nothing is gained, so `graph_proximity` = 1/(1+d) is 0 beyond 4. |
| `prefilter_max_dist` | 4 (changed from 3) | Same table: 3 → 4 still adds 5–12 points of true targets, and using `graph_max_dist` gives one value instead of two. |
| `recency_tau` | 100 commits | Gap between consecutive co-changes of the same pair: median 4 / 8 / 167, 75th percentile 8 / 49 / 566 commits (typhoon / vpc / govuk). τ = 100 (half-life ≈ 69 commits) lies within this range. f3 is normalised per query, so only the relative order of candidates matters, and its weight is learned. |
| `churn_window` | 200 commits | Gap between consecutive changes of the same block: median 5 / 14 / 82, 75th percentile 11 / 90 / 359. The window covers the typical re-change in all three repos. |
| Rule thresholds in features | none (min support 1, no min confidence) | Weak rules stay as low support / confidence values, and the GA weights them. Only 36 / 65 / 38 % (vpc / typhoon / govuk) of true targets have *any* earlier co-change with the seed, so history alone cannot reach most targets; the other feature families exist for that reason. |
| Add-type candidates | no cap | 15–22 per query (median), at most 95 observed. |
| Pre-filter trimming order | support, distance, same file, same dir, name sim, degree | Strongest evidence first; applied only when the union exceeds the budget. |
| Normalisation of `hist_cochange_count`, `hist_cochange_recency`, `graph_cand_degree`, `hist_cand_recent_changes` | log1p / per-query max / per-snapshot max / per-query max | Maps raw counts to [0, 1] per query, so features are comparable across repos of very different size and history length. |
| `text_content_similarity` weighting | tf = 1 + ln(count), idf = ln((1+N)/(1+df)) + 1, cosine | Standard TF-IDF (scikit-learn's smoothed, sublinear variant); no tunable threshold. Tokens = lower-case alphanumeric runs of the canonical body (keys, values, references); no stop-word list, IDF down-weights common tokens. |
| Adamic–Adar degree floor | 2 | A shared neighbour touches both blocks, so its degree is ≥ 2; the floor only affects the normaliser (seed neighbours used by the seed alone) and avoids ln(1) = 0. Not a tuning choice. |
| `graph_propagation_conf` cells | direct neighbours, action {+,~,−} × direction {in, out} | Distance 1 only: the rule describes direct dependence. Siblings at distance 2 are already covered by `graph_proximity` / `graph_shared_neighbors`, and more cells would split the (rarer) removal evidence further. |
| Model features | 12 (`model.features`), locality stored only | See SPEC §15.12: no measurable effect of the locality features in a proxy ablation; reversible without re-mining. |

Observation for the report: in `poseidon/typhoon`, 88 % of true targets are unreachable in the reference graph, because the repo keeps near-identical per-platform copies (`aws/…`, `azure/…`, `google-cloud/…`) and one change is applied to all copies. References never cross directories, so name similarity and history carry those cases. This shows why a fixed combination of signals fails across repo styles.


## GA (`ga.*`), decided in milestone 5

| Parameter | Value | Evidence / justification |
|---|---|---|
| `fitness_sample` | 5,000 queries (equal per repo) | One-fold check (`docs/RESULTS.md` §4): test MRR 0.701 ± 0.008 (2,000), **0.705 ± 0.000 (5,000)**, 0.706 ± 0.003 (all 43k queries) at 7 / 20 / 182 s per run. 5,000 matches all data at 1/9 of the cost. |
| Budget: population × generations | ≈ 3,600 evaluations | Convergence check (5 folds × 3 seeds × 200 generations at population 60): validation MRR 0.723 (gen 10), 0.735 (gen 60), 0.738 (gen 200); training fitness keeps creeping up while validation is flat, so more generations mostly fit the sample. |
| Population / generations, mutation, crossover | tuned per fold by grid search | 3 × 3 × 3 grid on validation repos (`docs/RESULTS.md` §5.5): all configurations within 0.004–0.010; low mutation (0.1) preferred in 4 of 5 folds. Final model: population 30, 120 generations, mutation 0.1, crossover 0.8. |
| Tournament 3, elitism 2, BLX-0.5, σ 0.2 → 0.02 | fixed | Standard values; the grid showed the GA is insensitive to its settings, so tuning them further is not warranted. |
| Seeds | 10 per fold | SD over seeds 0.0008 MRR (`ga`): more runs add no information. |
| Weights ≥ 0, sum 1; offset ∈ [−1, 1] | fixed | `ga_negative` check: same MRR (0.742 vs 0.746) and no feature with a negative mean weight. 15-gene variant with separate add-type weights: +0.002, within noise (§4.1). |

## Removed

| Parameter | Reason |
|---|---|
| `max_size_kb` | Blobless clones fetch only the `.tf` blobs needed, so repository size no longer affects cost. |
| `max_selected` | Redundant: every repo that passes the filters within the shortlist is kept. |
| `target_repos` | Unused; dataset size follows from the filters. |
