from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    terrads_db: Path = Path("data/external/terrads/TerraDS.sqlite")
    repos_csv: Path = Path("data/raw/repos.csv")
    shortlist_csv: Path = Path("data/raw/shortlist.csv")
    dataset_csv: Path = Path("data/raw/dataset.csv")
    tmp: Path = Path("data/tmp")
    processed: Path = Path("data/processed")
    outputs: Path = Path("data/results")  # everything runs write (gitignored)
    logs: Path = Path("data/logs")
    models: Path = Path("models")
    results: Path = Path("results")  # curated, committed result tables + figures


@dataclass
class SelectionConfig:
    provider: str = "aws"
    min_resources: int = 20
    min_stars: int = 0
    shortlist: int | None = 150
    min_tf_commits: int = 50
    min_usable_commits: int = 50
    max_add_only_frac: float = 0.4
    dedup_forks: bool = True
    workers: int = 6
    exclude_keywords: list[str] = field(default_factory=list)
    exclude_repos: list[str] = field(default_factory=list)


@dataclass
class CloneConfig:
    mode: str = "blobless_prefetch"
    timeout_s: int = 600
    prefetch_chunk: int = 2000


@dataclass
class MiningConfig:
    max_commits: int = 2000
    min_items: int = 2
    max_items: int = 30
    warmup_frac: float = 0.2
    use_merge_commits: bool = True


@dataclass
class FeaturesConfig:
    max_candidates: int = 200
    prefilter_max_dist: int = 4
    graph_max_dist: int = 4
    recency_tau: float = 100.0
    churn_window: int = 200


def _default_model_features() -> list[str]:
    return [
        "hist_cochange_conf",
        "hist_cochange_count",
        "hist_cochange_recency",
        "hist_cand_recent_changes",
        "hist_file_cochange_conf",
        "hist_type_cochange_conf",
        "graph_proximity",
        "graph_shared_neighbors",
        "graph_cand_degree",
        "graph_propagation_conf",
        "text_name_similarity",
        "text_content_similarity",
    ]


@dataclass
class ModelConfig:
    # columns of features.parquet the methods use; not part of settings_hash (no re-mine)
    features: list[str] = field(default_factory=_default_model_features)


@dataclass
class ArmConfig:
    min_support: int = 1
    min_confidence: float = 0.1
    top_k: int = 10


@dataclass
class BaselinesConfig:
    arm: ArmConfig = field(default_factory=ArmConfig)


@dataclass
class GridConfig:
    # (population, generations) pairs at about the same evaluation budget
    pop_gen: list[list[int]] = field(default_factory=lambda: [[30, 120], [60, 60], [120, 30]])
    mutation_rate: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    crossover_rate: list[float] = field(default_factory=lambda: [0.6, 0.8, 0.95])
    seeds: int = 3


@dataclass
class GAConfig:
    population: int = 60
    generations: int = 60
    tournament: int = 3
    crossover_rate: float = 0.8
    blend_alpha: float = 0.5
    mutation_rate: float = 0.3
    sigma_start: float = 0.2
    sigma_end: float = 0.02
    elitism: int = 2
    fitness_sample: int = 5000
    offset_range: float = 1.0
    check_seeds: int = 1
    convergence_generations: int = 200
    convergence_seeds: int = 3
    convergence_tol: float = 0.001
    valid_frac: float = 0.25
    tune: bool = True
    grid: GridConfig = field(default_factory=GridConfig)


@dataclass
class EvaluationConfig:
    k: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    folds: int = 5
    seeds: int = 10
    bootstrap: int = 1000
    methods: list[str] = field(
        default_factory=lambda: ["ga", "random_search", "arm", "fixed_weights", "random"]
    )


@dataclass
class ExperimentsConfig:
    seeds: int = 3  # GA runs per fold (ablation) or draws per fold and fraction (learning curve)
    ablation: list[str] = field(default_factory=lambda: ["hist", "graph", "text"])
    learning_curve: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75])


@dataclass
class RecommendConfig:
    top: int = 10
    combine: str = "max"  # several seeds: "max" (related to any edit) or "sum"
    explain_top: int = 3  # contributions shown per suggestion with --explain


@dataclass
class OutputConfig:
    csv: bool = True
    csv_features: bool = False


@dataclass
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    clone: CloneConfig = field(default_factory=CloneConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    recommend: RecommendConfig = field(default_factory=RecommendConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for name, value in data.items():
        current = getattr(defaults, name)
        if is_dataclass(current):
            kwargs[name] = _build(type(current), value or {})
        elif isinstance(current, Path):
            kwargs[name] = Path(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Path | str | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Plain-dict view (Paths as str) for recording alongside results."""
    from dataclasses import asdict

    def conv(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v

    return conv(asdict(cfg))
