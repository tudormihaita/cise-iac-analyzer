"""Self-implemented genetic algorithm learning the ranking weights (numpy only).

data.py       training queries, repo-stratified sampling, label permutation
fitness.py    scoring and the vectorized MRR fitness
operators.py  repair, random individuals, tournament, blend crossover, Gaussian mutation
search.py     the GA loop (`evolve`) and random search with the same budget
grid.py       nested grid search over GA settings
runs.py       reproducible random streams, resumable run files
method.py     `GAMethod`, the evaluation method and its variants
"""

from tfimpact.methods.ga.data import Batch, TrainSet
from tfimpact.methods.ga.fitness import fitness, scores
from tfimpact.methods.ga.grid import config_by_id, config_id, grid_configs
from tfimpact.methods.ga.method import VARIANTS, GAMethod
from tfimpact.methods.ga.operators import random_individual, repair
from tfimpact.methods.ga.runs import load_run, run_rng, save_run, settings_hash
from tfimpact.methods.ga.search import EvolveResult, evolve, random_search

__all__ = [
    "VARIANTS",
    "Batch",
    "EvolveResult",
    "GAMethod",
    "TrainSet",
    "config_by_id",
    "config_id",
    "evolve",
    "fitness",
    "grid_configs",
    "load_run",
    "random_individual",
    "random_search",
    "repair",
    "run_rng",
    "save_run",
    "scores",
    "settings_hash",
]
