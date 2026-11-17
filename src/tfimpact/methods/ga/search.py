"""The search loops: the genetic algorithm and random search with the same budget.

GA, per generation: keep the `elitism` best unchanged; fill the rest of the population with
children (tournament selection of two parents, blend crossover with probability
`crossover_rate`, Gaussian mutation with decaying sigma, repair); evaluate the new population.
Random search draws `population` random chromosomes per generation and keeps the best one, so
both use population x (generations + 1) fitness evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tfimpact.config import GAConfig
from tfimpact.methods.ga.data import Batch
from tfimpact.methods.ga.fitness import fitness
from tfimpact.methods.ga.operators import (
    blend_crossover,
    gaussian_mutation,
    linear_sigma,
    random_individual,
    repair,
    tournament,
)


@dataclass
class EvolveResult:
    best: np.ndarray
    best_fitness: float
    history: list[dict] = field(default_factory=list)  # per generation: best / mean / extra
    evaluations: int = 0


def evolve(
    batch: Batch,
    cfg: GAConfig,
    rng: np.random.Generator,
    n_features: int,
    negative: bool = False,
    generations: int | None = None,
    on_generation: Callable[[int, np.ndarray], dict] | None = None,
) -> EvolveResult:
    """Run the GA; `on_generation(g, best)` may add values to the per-generation history."""
    gens = cfg.generations if generations is None else generations
    n = cfg.population
    pop = np.array(
        [random_individual(rng, n_features, cfg.offset_range, negative) for _ in range(n)]
    )
    fit = fitness(pop, batch)
    evals = n
    history = [_record(0, pop, fit, on_generation)]
    for g in range(1, gens + 1):
        sigma = linear_sigma(cfg.sigma_start, cfg.sigma_end, g, gens)
        elite = pop[np.argsort(-fit, kind="stable")[: cfg.elitism]]
        children = []
        while len(children) < n - len(elite):
            a = pop[tournament(fit, cfg.tournament, rng)]
            b = pop[tournament(fit, cfg.tournament, rng)]
            if rng.random() < cfg.crossover_rate:
                a = blend_crossover(a, b, cfg.blend_alpha, rng)
            a = gaussian_mutation(a, cfg.mutation_rate, sigma, rng)
            children.append(repair(a, cfg.offset_range, negative))
        pop = np.vstack([elite, *children]) if len(elite) else np.array(children)
        fit = fitness(pop, batch)
        evals += n
        history.append(_record(g, pop, fit, on_generation))
    i = int(np.argmax(fit))
    return EvolveResult(pop[i], float(fit[i]), history, evals)


def random_search(
    batch: Batch,
    cfg: GAConfig,
    rng: np.random.Generator,
    n_features: int,
    generations: int | None = None,
) -> EvolveResult:
    """Same evaluation budget as `evolve` (population x (generations + 1)), no search operators."""
    gens = cfg.generations if generations is None else generations
    best, best_fit, history = None, -np.inf, []
    for g in range(gens + 1):
        cand = np.array(
            [
                random_individual(rng, n_features, cfg.offset_range, False)
                for _ in range(cfg.population)
            ]
        )
        fit = fitness(cand, batch)
        i = int(np.argmax(fit))
        if fit[i] > best_fit:
            best, best_fit = cand[i], float(fit[i])
        history.append({"gen": g, "best": best_fit, "mean": float(fit.mean())})
    assert best is not None
    return EvolveResult(best, best_fit, history, cfg.population * (gens + 1))


def _record(g: int, pop: np.ndarray, fit: np.ndarray, cb: Callable | None) -> dict:
    rec = {"gen": g, "best": float(fit.max()), "mean": float(fit.mean())}
    if cb is not None:
        rec.update(cb(g, pop[int(np.argmax(fit))]))
    return rec
