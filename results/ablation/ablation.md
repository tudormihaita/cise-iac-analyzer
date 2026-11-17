# Ablation

Core repos, mean over repos; per fold the grid-chosen configuration, 3 seeds. Paired tests full vs variant on per-repo MRR (Wilcoxon + Holm); RI = (full − variant) / variant.

| variant | targets | features | MRR | CI low | CI high | MAP | Hit@5 | full − variant | RI full | full better | variant better | p (Holm) | A12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ga | all | 12 | 0.745 | 0.727 | 0.766 | 0.436 | 0.846 |  |  |  |  |  |  |
| ga_hist | all | 6 | 0.660 | 0.635 | 0.685 | 0.379 | 0.762 | 0.085 | +12.9% | 53 | 0 | 7.2e-10 | 0.758 |
| ga_graph | all | 4 | 0.420 | 0.392 | 0.450 | 0.211 | 0.543 | 0.326 | +77.7% | 53 | 0 | 7.2e-10 | 0.995 |
| ga_text | all | 2 | 0.488 | 0.445 | 0.533 | 0.303 | 0.576 | 0.258 | +52.9% | 53 | 0 | 7.2e-10 | 0.916 |
| ga | existing | 12 | 0.648 | 0.619 | 0.678 | 0.437 | 0.753 |  |  |  |  |  |  |
| ga_hist | existing | 6 | 0.497 | 0.461 | 0.534 | 0.346 | 0.604 | 0.152 | +30.5% | 51 | 2 | 8e-10 | 0.813 |
| ga_graph | existing | 4 | 0.460 | 0.432 | 0.491 | 0.263 | 0.582 | 0.188 | +40.9% | 50 | 3 | 1.1e-09 | 0.895 |
| ga_text | existing | 2 | 0.533 | 0.494 | 0.572 | 0.362 | 0.627 | 0.115 | +21.6% | 52 | 1 | 8e-10 | 0.760 |
