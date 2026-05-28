from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class BenchmarkScenario:
    slug: str
    title: str
    description: str
    research_question: str
    evaluation_focus: tuple[str, ...]
    expected_behavior: str


SCENARIOS: dict[str, BenchmarkScenario] = {
    "strong-base": BenchmarkScenario(
        slug="strong-base",
        title="Strong-Base Scenario",
        description="The base table already contains most of the predictive signal, so augmentation should be judged by robustness, compactness, and safety rather than raw gain alone.",
        research_question="Can AutoFeatPlus avoid unnecessary or risky augmentation when the base table is already highly predictive?",
        evaluation_focus=("r2", "rmse", "mae", "n_features", "n_blocked_features"),
        expected_behavior="BASE is already strong; JOIN_ALL adds little; AutoFeatPlus should compress or de-risk features with limited accuracy loss.",
    ),
    "missing-signal": BenchmarkScenario(
        slug="missing-signal",
        title="Missing-Signal Augmentation Scenario",
        description="The base table is incomplete and useful predictive signal exists in external tables.",
        research_question="Can augmentation recover missing predictive signal from external tables while using fewer features than exhaustive joining?",
        evaluation_focus=("r2", "rmse", "mae", "delta_vs_base", "feature_reduction"),
        expected_behavior="JOIN_ALL and AutoFeat should significantly outperform BASE; AutoFeatPlus should approach JOIN_ALL with fewer features.",
    ),
    "wide-multi-table": BenchmarkScenario(
        slug="wide-multi-table",
        title="Wide Multi-Table Scenario",
        description="Joining many related tables creates a wide feature space with redundancy and noise.",
        research_question="Can AutoFeatPlus retain the useful subset of a wide joined feature space while reducing redundancy and privacy risk?",
        evaluation_focus=("r2", "rmse", "mae", "n_features", "feature_selection_time"),
        expected_behavior="JOIN_ALL may be strong but wide; selection quality matters more than pure access to all features.",
    ),
    "privacy-proxy-sensitive": BenchmarkScenario(
        slug="privacy-proxy-sensitive",
        title="Privacy / Proxy-Sensitive Scenario",
        description="Some highly predictive features are sensitive identifiers or target proxies, so the benchmark must examine utility-privacy trade-offs.",
        research_question="Can AutoFeatPlus preserve useful signal while avoiding sensitive identifiers and target-proxy shortcuts?",
        evaluation_focus=("r2", "rmse", "mae", "n_blocked_features", "privacy_risk_score"),
        expected_behavior="Naive baselines may achieve the highest scores; AutoFeatPlus should trade some utility for more realistic and safer feature sets.",
    ),
    "weak-join-alignment": BenchmarkScenario(
        slug="weak-join-alignment",
        title="Weak-Join / Alignment-Limited Scenario",
        description="External tables exist but temporal or key alignment is weak, so augmentation is limited by join quality.",
        research_question="How robust is augmentation when join quality is the primary bottleneck?",
        evaluation_focus=("joined_feature_count", "all_null_joined_features", "mean_missing_ratio_joined", "r2"),
        expected_behavior="JOIN_ALL shows little or no gain; diagnostics reveal sparse or null joined features; tuning alignment matters more than model choice.",
    ),
    "high-dimensional-structured": BenchmarkScenario(
        slug="high-dimensional-structured",
        title="High-Dimensional Structured Augmentation Scenario",
        description="Augmentation produces high-dimensional structured features such as CSI tensors, so both feature selection and downstream representation matter.",
        research_question="Can AutoFeatPlus compress structured high-dimensional features without destroying downstream model usability?",
        evaluation_focus=("r2", "accuracy", "macro_f1", "n_features", "train_time"),
        expected_behavior="Full structured inputs may perform best; compact selected features help only if the downstream representation remains compatible with the model family.",
    ),
    "generalization-critical": BenchmarkScenario(
        slug="generalization-critical",
        title="Generalization-Critical Scenario",
        description="Random splits may look strong, but the real question is how augmentation behaves on unseen users, positions, or future time periods.",
        research_question="Do augmented features generalize beyond the users, positions, or time ranges seen during training?",
        evaluation_focus=("r2", "accuracy", "split_mode", "test_groups"),
        expected_behavior="Random split can be optimistic; holdout splits reveal whether the learned signal generalizes or only interpolates.",
    ),
}


BASE_TABLE_USE_CASES: dict[str, BenchmarkScenario] = {
    "temporal-anchor": BenchmarkScenario(
        slug="temporal-anchor",
        title="Temporal Anchor Base Table",
        description="The base table is naturally indexed by time and should retrieve related tables through temporal alignment.",
        research_question="Can the pipeline find temporally aligned augmentation candidates around the user-provided base table?",
        evaluation_focus=("join_mode", "join_key", "time_tolerance_seconds", "match_ratio"),
        expected_behavior="The planner should prefer time-based asof joins and estimate a useful tolerance window.",
    ),
    "configuration-anchor": BenchmarkScenario(
        slug="configuration-anchor",
        title="Configuration Anchor Base Table",
        description="The base table is keyed by workload or deployment settings such as n, c, cpu_limit, or memory limits.",
        research_question="Can the pipeline discover configuration-level relationships when row-level temporal alignment is weak?",
        evaluation_focus=("join_key", "config_overlap", "confidence"),
        expected_behavior="The planner should surface exact or grouped joins over shared experimental settings.",
    ),
    "privacy-sensitive-base": BenchmarkScenario(
        slug="privacy-sensitive-base",
        title="Privacy-Sensitive Base Table",
        description="The base table contains columns that can directly reveal identifiers, location, or target-proxy signals.",
        research_question="Can the pipeline recognize that augmentation must be filtered through privacy-aware policies for this base table?",
        evaluation_focus=("privacy_policy", "blocked_features", "privacy_risk_score"),
        expected_behavior="The planner should flag the task as privacy-sensitive and encourage AutoFeatPlus-style filtering.",
    ),
    "high-dimensional-base": BenchmarkScenario(
        slug="high-dimensional-base",
        title="High-Dimensional Base Table",
        description="The base table is already wide or structured, so augmentation should preserve structure and avoid redundant explosion.",
        research_question="Can the pipeline keep the base table usable for structured downstream models after augmentation?",
        evaluation_focus=("n_features", "representation", "downstream_model_family"),
        expected_behavior="The pipeline should emphasize compactness and representation-aware augmentation.",
    ),
    "cross-application-bridge": BenchmarkScenario(
        slug="cross-application-bridge",
        title="Cross-Application Bridge Base Table",
        description="The base table is expected to pull useful signal from tables belonging to different applications or services.",
        research_question="Can the pipeline retrieve useful cross-application tables while rejecting weak semantic matches?",
        evaluation_focus=("candidate_tables", "join_diagnostics", "r2"),
        expected_behavior="The planner should separate true joinable candidates from merely similar measurement tables.",
    ),
}


def infer_benchmark_scenarios(
    *,
    data_label: str = "",
    source_file: str = "",
    split_mode: str = "",
    target_column: str = "",
) -> list[str]:
    label = data_label.lower()
    source = source_file.lower()
    split = split_mode.lower()
    target = target_column.lower()
    scenarios: list[str] = []

    if "scenario1_rabbitmq" in label:
        scenarios.append("strong-base")
    if "scenario2c_rabbitmq_reduced" in label:
        scenarios.append("missing-signal")
    if "scenario3_amf_seg01" in label:
        scenarios.append("wide-multi-table")

    is_kul = "kul" in label or "kul" in source
    is_eur = "eur" in label or "/eur/" in source or "6907619" in source
    is_cleaned_eur = "cleaned" in label or "cleaned" in source

    if is_kul and "with_metadata" in source:
        scenarios.append("privacy-proxy-sensitive")
    if is_kul and split in {"random", ""} and "with_metadata" not in source and "downstream_models" not in source:
        scenarios.append("high-dimensional-structured")
    if is_kul and ("holdout" in split or "holdout" in source):
        scenarios.append("generalization-critical")
    if is_kul and "downstream_models" in source:
        scenarios.append("high-dimensional-structured")

    if is_eur:
        scenarios.extend(["strong-base", "privacy-proxy-sensitive"])
    if is_cleaned_eur:
        scenarios.append("weak-join-alignment")
    elif is_eur and "downstream_models" in source:
        scenarios.append("wide-multi-table")

    if target in {"lat99", "lat99_ms"} and "privacy-proxy-sensitive" not in scenarios:
        scenarios.append("privacy-proxy-sensitive")

    # Preserve order and uniqueness.
    seen: set[str] = set()
    ordered: list[str] = []
    for scenario in scenarios:
        if scenario not in seen:
            ordered.append(scenario)
            seen.add(scenario)
    return ordered


def scenario_titles(slugs: list[str]) -> str:
    return " | ".join(SCENARIOS[slug].title for slug in slugs if slug in SCENARIOS)


def scenario_markdown() -> str:
    lines = ["# Benchmark Scenarios", ""]
    for scenario in SCENARIOS.values():
        lines.append(f"## {scenario.title}")
        lines.append(scenario.description)
        lines.append("")
        lines.append(f"Research question: {scenario.research_question}")
        lines.append("")
        lines.append(f"Evaluation focus: {', '.join(scenario.evaluation_focus)}")
        lines.append("")
        lines.append(f"Expected behavior: {scenario.expected_behavior}")
        lines.append("")
    return "\n".join(lines)


def infer_base_table_use_cases(
    *,
    base_table: str,
    columns: list[str],
    target_column: str = "",
    metadata_text: str = "",
) -> list[str]:
    base_name = base_table.lower()
    normalized_columns = {column.lower() for column in columns}
    target = target_column.lower()
    metadata = metadata_text.lower()
    use_cases: list[str] = []

    if "time" in normalized_columns:
        use_cases.append("temporal-anchor")

    if {"n", "c", "cpu_limit", "ram_limit", "ram_limit_mb", "cpu_limit_mb"} & normalized_columns:
        use_cases.append("configuration-anchor")

    if re.search(r"user|sample|device|location|target_[xyz]|lat99|lat95|lat75|lat50|min|mean", " ".join(normalized_columns)) or target in {"lat99", "lat99_ms", "target_x"}:
        use_cases.append("privacy-sensitive-base")

    if len(columns) >= 40 or any("subcarrier" in column.lower() or "antenna_" in column.lower() for column in columns):
        use_cases.append("high-dimensional-base")

    if any(keyword in base_name for keyword in ["rabbitmq", "amf", "python", "golang"]) or any(
        keyword in metadata for keyword in ["web server", "rabbitmq", "amf", "5g core"]
    ):
        use_cases.append("cross-application-bridge")

    seen: set[str] = set()
    ordered: list[str] = []
    for use_case in use_cases:
        if use_case not in seen:
            ordered.append(use_case)
            seen.add(use_case)
    return ordered


def use_case_titles(slugs: list[str]) -> str:
    return " | ".join(BASE_TABLE_USE_CASES[slug].title for slug in slugs if slug in BASE_TABLE_USE_CASES)


def base_table_use_case_markdown() -> str:
    lines = ["# Base-Table Use Cases", ""]
    for scenario in BASE_TABLE_USE_CASES.values():
        lines.append(f"## {scenario.title}")
        lines.append(scenario.description)
        lines.append("")
        lines.append(f"Research question: {scenario.research_question}")
        lines.append("")
        lines.append(f"Evaluation focus: {', '.join(scenario.evaluation_focus)}")
        lines.append("")
        lines.append(f"Expected behavior: {scenario.expected_behavior}")
        lines.append("")
    return "\n".join(lines)
