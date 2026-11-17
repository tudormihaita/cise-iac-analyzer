# Evaluation (53 core repos, grouped 5-fold CV; 8 add-only repos held out)

Mean over repos (each repo weighted equally); MRR as mean ± SD over repos [bootstrap 95% CI]. Subsets: core = cross-validated repos (headline), add_only = never trained on, all = both. Other metrics' SDs/CIs and SD over folds: metrics.csv.

## Targets: all

| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|
| ga | core | 0.746 ± 0.076 [0.727, 0.766] | 0.437 | 0.665 | 0.846 | 0.455 | 0.439 |
| ga | add_only | 0.716 ± 0.173 [0.602, 0.809] | 0.482 | 0.660 | 0.780 | 0.511 | 0.494 |
| ga | all | 0.742 ± 0.092 [0.716, 0.765] | 0.443 | 0.664 | 0.838 | 0.462 | 0.446 |
| random_search | core | 0.733 ± 0.079 [0.713, 0.754] | 0.430 | 0.650 | 0.835 | 0.448 | 0.430 |
| random_search | add_only | 0.713 ± 0.171 [0.600, 0.805] | 0.482 | 0.656 | 0.780 | 0.512 | 0.494 |
| random_search | all | 0.730 ± 0.094 [0.704, 0.754] | 0.437 | 0.651 | 0.827 | 0.456 | 0.438 |
| arm | core | 0.481 ± 0.152 [0.443, 0.522] | 0.231 | 0.426 | 0.555 | 0.283 | 0.267 |
| arm | add_only | 0.667 ± 0.162 [0.559, 0.759] | 0.392 | 0.620 | 0.729 | 0.447 | 0.463 |
| arm | all | 0.506 ± 0.164 [0.467, 0.545] | 0.252 | 0.452 | 0.578 | 0.305 | 0.293 |
| fixed_weights | core | 0.563 ± 0.147 [0.526, 0.601] | 0.332 | 0.473 | 0.670 | 0.344 | 0.337 |
| fixed_weights | add_only | 0.177 ± 0.118 [0.104, 0.250] | 0.103 | 0.130 | 0.196 | 0.090 | 0.094 |
| fixed_weights | all | 0.513 ± 0.194 [0.467, 0.562] | 0.302 | 0.428 | 0.608 | 0.311 | 0.305 |
| random | core | 0.118 ± 0.037 [0.108, 0.128] | 0.053 | 0.038 | 0.162 | 0.043 | 0.037 |
| random | add_only | 0.102 ± 0.041 [0.077, 0.129] | 0.048 | 0.033 | 0.137 | 0.044 | 0.035 |
| random | all | 0.116 ± 0.038 [0.107, 0.126] | 0.053 | 0.037 | 0.159 | 0.043 | 0.036 |

## Targets: existing

| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|
| ga | core | 0.649 ± 0.110 [0.622, 0.680] | 0.439 | 0.561 | 0.755 | 0.475 | 0.389 |
| ga | add_only | 0.671 ± 0.105 [0.599, 0.735] | 0.427 | 0.589 | 0.768 | 0.462 | 0.370 |
| ga | all | 0.652 ± 0.109 [0.624, 0.680] | 0.437 | 0.565 | 0.757 | 0.473 | 0.386 |
| random_search | core | 0.644 ± 0.110 [0.616, 0.674] | 0.438 | 0.552 | 0.751 | 0.474 | 0.384 |
| random_search | add_only | 0.655 ± 0.102 [0.587, 0.718] | 0.424 | 0.570 | 0.755 | 0.461 | 0.364 |
| random_search | all | 0.645 ± 0.109 [0.617, 0.673] | 0.436 | 0.555 | 0.752 | 0.473 | 0.382 |
| arm | core | 0.360 ± 0.164 [0.318, 0.405] | 0.211 | 0.301 | 0.440 | 0.278 | 0.222 |
| arm | add_only | 0.268 ± 0.167 [0.168, 0.382] | 0.178 | 0.228 | 0.319 | 0.229 | 0.168 |
| arm | all | 0.348 ± 0.166 [0.311, 0.390] | 0.207 | 0.292 | 0.424 | 0.271 | 0.215 |
| fixed_weights | core | 0.644 ± 0.106 [0.617, 0.673] | 0.444 | 0.542 | 0.769 | 0.490 | 0.384 |
| fixed_weights | add_only | 0.630 ± 0.129 [0.553, 0.719] | 0.424 | 0.549 | 0.731 | 0.469 | 0.356 |
| fixed_weights | all | 0.642 ± 0.108 [0.614, 0.671] | 0.442 | 0.543 | 0.764 | 0.487 | 0.380 |
| random | core | 0.116 ± 0.037 [0.107, 0.126] | 0.058 | 0.037 | 0.160 | 0.050 | 0.037 |
| random | add_only | 0.135 ± 0.090 [0.088, 0.204] | 0.083 | 0.041 | 0.202 | 0.089 | 0.061 |
| random | all | 0.118 ± 0.046 [0.108, 0.131] | 0.062 | 0.038 | 0.166 | 0.055 | 0.040 |

Candidate recall ceiling (mean over repos): core 0.810, add_only 0.671, all 0.792

Configuration chosen by the grid search (`ga`): fold 0: p30g120_m0.1_c0.8; fold 1: p60g60_m0.1_c0.8; fold 2: p30g120_m0.1_c0.8; fold 3: p30g120_m0.3_c0.95; fold 4: p30g120_m0.1_c0.95

SD over runs (GA seeds), core MRR: ga (all) 0.0008, ga (existing) 0.0030, random_search (all) 0.0031, random_search (existing) 0.0033

## Paired comparisons (core repos, per-repo MRR, all targets)

| A vs B | mean A | mean B | A better / B better | p (Holm) | A12 |
|---|---|---|---|---|---|
| ga vs random_search | 0.746 | 0.733 | 50 / 3 | 2.4e-09 | 0.56 (negligible) |
| ga vs arm | 0.746 | 0.481 | 53 / 0 | 2.4e-09 | 0.93 (large) |
| ga vs fixed_weights | 0.746 | 0.563 | 51 / 2 | 2.4e-09 | 0.87 (large) |
| ga vs random | 0.746 | 0.118 | 53 / 0 | 2.4e-09 | 1.00 (large) |
| random_search vs arm | 0.733 | 0.481 | 53 / 0 | 2.4e-09 | 0.92 (large) |
| random_search vs fixed_weights | 0.733 | 0.563 | 51 / 2 | 2.4e-09 | 0.85 (large) |
| random_search vs random | 0.733 | 0.118 | 53 / 0 | 2.4e-09 | 1.00 (large) |
| arm vs fixed_weights | 0.481 | 0.563 | 14 / 39 | 0.0015 | 0.34 (medium) |
| arm vs random | 0.481 | 0.118 | 53 / 0 | 2.4e-09 | 1.00 (large) |
| fixed_weights vs random | 0.563 | 0.118 | 53 / 0 | 2.4e-09 | 1.00 (large) |
