# Evaluation (53 core repos, grouped 5-fold CV; 8 add-only repos held out)

Mean over repos (each repo weighted equally); MRR as mean ± SD over repos [bootstrap 95% CI]. Subsets: core = cross-validated repos (headline), add_only = never trained on, all = both. Other metrics' SDs/CIs and SD over folds: metrics.csv.

## Targets: all

| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|
| ga_permuted | core | 0.428 ± 0.181 [0.380, 0.476] | 0.232 | 0.323 | 0.543 | 0.248 | 0.229 |
| ga_permuted | add_only | 0.227 ± 0.072 [0.182, 0.270] | 0.150 | 0.181 | 0.271 | 0.156 | 0.150 |
| ga_permuted | all | 0.401 ± 0.184 [0.354, 0.450] | 0.221 | 0.304 | 0.507 | 0.236 | 0.219 |
| ga_negative | core | 0.742 ± 0.079 [0.721, 0.763] | 0.432 | 0.663 | 0.839 | 0.446 | 0.433 |
| ga_negative | add_only | 0.716 ± 0.171 [0.603, 0.809] | 0.485 | 0.660 | 0.782 | 0.514 | 0.496 |
| ga_negative | all | 0.739 ± 0.094 [0.713, 0.763] | 0.439 | 0.662 | 0.832 | 0.455 | 0.441 |
| fixed_weights | core | 0.563 ± 0.147 [0.526, 0.601] | 0.332 | 0.473 | 0.670 | 0.344 | 0.337 |
| fixed_weights | add_only | 0.177 ± 0.118 [0.104, 0.250] | 0.103 | 0.130 | 0.196 | 0.090 | 0.094 |
| fixed_weights | all | 0.513 ± 0.194 [0.467, 0.562] | 0.302 | 0.428 | 0.608 | 0.311 | 0.305 |

## Targets: existing

| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|
| ga_permuted | core | 0.461 ± 0.172 [0.417, 0.508] | 0.301 | 0.343 | 0.594 | 0.343 | 0.249 |
| ga_permuted | add_only | 0.415 ± 0.112 [0.348, 0.495] | 0.290 | 0.314 | 0.527 | 0.335 | 0.243 |
| ga_permuted | all | 0.455 ± 0.165 [0.412, 0.499] | 0.299 | 0.339 | 0.585 | 0.342 | 0.248 |
| ga_negative | core | 0.641 ± 0.117 [0.609, 0.673] | 0.437 | 0.555 | 0.740 | 0.470 | 0.385 |
| ga_negative | add_only | 0.685 ± 0.109 [0.610, 0.751] | 0.450 | 0.607 | 0.777 | 0.484 | 0.378 |
| ga_negative | all | 0.647 ± 0.116 [0.616, 0.677] | 0.439 | 0.562 | 0.745 | 0.472 | 0.384 |
| fixed_weights | core | 0.644 ± 0.106 [0.617, 0.673] | 0.444 | 0.542 | 0.769 | 0.490 | 0.384 |
| fixed_weights | add_only | 0.630 ± 0.129 [0.553, 0.719] | 0.424 | 0.549 | 0.731 | 0.469 | 0.356 |
| fixed_weights | all | 0.642 ± 0.108 [0.614, 0.671] | 0.442 | 0.543 | 0.764 | 0.487 | 0.380 |

Candidate recall ceiling (mean over repos): core 0.810, add_only 0.671, all 0.792

## Paired comparisons (core repos, per-repo MRR, all targets)

| A vs B | mean A | mean B | A better / B better | p (Holm) | A12 |
|---|---|---|---|---|---|
| ga_permuted vs ga_negative | 0.428 | 0.742 | 1 / 52 | 7.6e-10 | 0.05 (large) |
| ga_permuted vs fixed_weights | 0.428 | 0.563 | 3 / 50 | 9.9e-09 | 0.28 (large) |
| ga_negative vs fixed_weights | 0.742 | 0.563 | 50 / 3 | 8.9e-10 | 0.86 (large) |
