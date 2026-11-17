# Evaluation (53 core repos, grouped 5-fold CV; 8 add-only repos held out)

Mean over repos (each repo weighted equally); MRR as mean ± SD over repos [bootstrap 95% CI]. Subsets: core = cross-validated repos (headline), add_only = never trained on, all = both. Other metrics' SDs/CIs and SD over folds: metrics.csv.

## Targets: all

| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|
| ga | core | 0.745 ± 0.076 [0.727, 0.766] | 0.436 | 0.665 | 0.846 | 0.455 | 0.439 |
| ga | add_only | 0.716 ± 0.173 [0.602, 0.809] | 0.482 | 0.660 | 0.781 | 0.510 | 0.494 |
| ga | all | 0.742 ± 0.092 [0.716, 0.765] | 0.442 | 0.665 | 0.837 | 0.462 | 0.446 |
| ga_hist | core | 0.660 ± 0.097 [0.635, 0.685] | 0.379 | 0.574 | 0.762 | 0.395 | 0.373 |
| ga_hist | add_only | 0.695 ± 0.170 [0.580, 0.789] | 0.473 | 0.641 | 0.758 | 0.498 | 0.481 |
| ga_hist | all | 0.665 ± 0.108 [0.637, 0.691] | 0.391 | 0.583 | 0.762 | 0.409 | 0.387 |
| ga_graph | core | 0.420 ± 0.112 [0.392, 0.450] | 0.211 | 0.303 | 0.543 | 0.234 | 0.214 |
| ga_graph | add_only | 0.223 ± 0.086 [0.169, 0.276] | 0.183 | 0.103 | 0.370 | 0.215 | 0.155 |
| ga_graph | all | 0.394 ± 0.127 [0.366, 0.424] | 0.207 | 0.277 | 0.520 | 0.231 | 0.206 |
| ga_text | core | 0.488 ± 0.162 [0.445, 0.533] | 0.303 | 0.395 | 0.576 | 0.312 | 0.290 |
| ga_text | add_only | 0.175 ± 0.071 [0.129, 0.223] | 0.133 | 0.092 | 0.232 | 0.141 | 0.087 |
| ga_text | all | 0.447 ± 0.186 [0.402, 0.495] | 0.281 | 0.356 | 0.531 | 0.289 | 0.263 |

## Targets: existing

| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|
| ga | core | 0.648 ± 0.112 [0.619, 0.678] | 0.437 | 0.560 | 0.753 | 0.474 | 0.387 |
| ga | add_only | 0.671 ± 0.107 [0.598, 0.735] | 0.427 | 0.589 | 0.767 | 0.462 | 0.370 |
| ga | all | 0.651 ± 0.110 [0.622, 0.679] | 0.436 | 0.563 | 0.755 | 0.472 | 0.385 |
| ga_hist | core | 0.497 ± 0.142 [0.461, 0.534] | 0.346 | 0.401 | 0.604 | 0.378 | 0.293 |
| ga_hist | add_only | 0.402 ± 0.240 [0.256, 0.566] | 0.300 | 0.331 | 0.468 | 0.328 | 0.244 |
| ga_hist | all | 0.484 ± 0.159 [0.445, 0.524] | 0.340 | 0.391 | 0.586 | 0.371 | 0.287 |
| ga_graph | core | 0.460 ± 0.114 [0.432, 0.491] | 0.263 | 0.350 | 0.582 | 0.307 | 0.233 |
| ga_graph | add_only | 0.487 ± 0.096 [0.427, 0.554] | 0.299 | 0.419 | 0.564 | 0.326 | 0.304 |
| ga_graph | all | 0.463 ± 0.111 [0.436, 0.491] | 0.268 | 0.359 | 0.580 | 0.309 | 0.242 |
| ga_text | core | 0.533 ± 0.145 [0.494, 0.572] | 0.362 | 0.445 | 0.627 | 0.388 | 0.319 |
| ga_text | add_only | 0.484 ± 0.160 [0.378, 0.585] | 0.286 | 0.377 | 0.603 | 0.349 | 0.256 |
| ga_text | all | 0.527 ± 0.147 [0.491, 0.564] | 0.352 | 0.436 | 0.624 | 0.383 | 0.311 |

Candidate recall ceiling (mean over repos): core 0.810, add_only 0.671, all 0.792

Configuration chosen by the grid search (`ga`): fold 0: p30g120_m0.1_c0.8; fold 0: p30g120_m0.1_c0.8; fold 0: p30g120_m0.1_c0.8; fold 0: p30g120_m0.1_c0.8; fold 1: p60g60_m0.1_c0.8; fold 1: p60g60_m0.1_c0.8; fold 1: p60g60_m0.1_c0.8; fold 1: p60g60_m0.1_c0.8; fold 2: p30g120_m0.1_c0.8; fold 2: p30g120_m0.1_c0.8; fold 2: p30g120_m0.1_c0.8; fold 2: p30g120_m0.1_c0.8; fold 3: p30g120_m0.3_c0.95; fold 3: p30g120_m0.3_c0.95; fold 3: p30g120_m0.3_c0.95; fold 3: p30g120_m0.3_c0.95; fold 4: p30g120_m0.1_c0.95; fold 4: p30g120_m0.1_c0.95; fold 4: p30g120_m0.1_c0.95; fold 4: p30g120_m0.1_c0.95

SD over runs (GA seeds), core MRR: ga (all) 0.0004, ga (existing) 0.0008, ga_graph (all) 0.0010, ga_graph (existing) 0.0003, ga_hist (all) 0.0010, ga_hist (existing) 0.0016, ga_text (all) 0.0017, ga_text (existing) 0.0006

## Paired comparisons (core repos, per-repo MRR, all targets)

| A vs B | mean A | mean B | RI of A | A better / B better | p (Holm) | A12 |
|---|---|---|---|---|---|---|
| ga vs ga_hist | 0.745 | 0.660 | +12.9% | 53 / 0 | 7.2e-10 | 0.76 (large) |
| ga vs ga_graph | 0.745 | 0.420 | +77.7% | 53 / 0 | 7.2e-10 | 1.00 (large) |
| ga vs ga_text | 0.745 | 0.488 | +52.9% | 53 / 0 | 7.2e-10 | 0.92 (large) |
