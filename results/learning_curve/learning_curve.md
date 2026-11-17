# Learning curve

Core test repos, mean over repos; training repos subsampled per fold (3 nested draws per share); 100 % = main run (seeds 0–2). Training MRR = fitness on the run's training sample.

| share | training repos | targets | test MRR | CI low | CI high | SD draws | SD folds | Hit@5 | training MRR |
|---|---|---|---|---|---|---|---|---|---|
| 10% | 4.0 | all | 0.734 | 0.714 | 0.754 | 0.002 | 0.039 | 0.836 | 0.746 |
| 25% | 10.4 | all | 0.743 | 0.723 | 0.764 | 0.002 | 0.033 | 0.844 | 0.751 |
| 50% | 21.4 | all | 0.747 | 0.728 | 0.767 | 0.002 | 0.031 | 0.847 | 0.752 |
| 75% | 32.0 | all | 0.744 | 0.725 | 0.765 | 0.001 | 0.031 | 0.845 | 0.748 |
| 100% | 42.4 | all | 0.745 | 0.727 | 0.766 | 0.000 | 0.032 | 0.846 | 0.748 |
| 10% | 4.0 | existing | 0.641 | 0.612 | 0.672 | 0.002 | 0.054 | 0.747 | 0.746 |
| 25% | 10.4 | existing | 0.648 | 0.620 | 0.678 | 0.008 | 0.052 | 0.754 | 0.751 |
| 50% | 21.4 | existing | 0.652 | 0.623 | 0.683 | 0.001 | 0.050 | 0.757 | 0.752 |
| 75% | 32.0 | existing | 0.648 | 0.621 | 0.678 | 0.002 | 0.051 | 0.754 | 0.748 |
| 100% | 42.4 | existing | 0.648 | 0.619 | 0.678 | 0.001 | 0.054 | 0.753 | 0.748 |
