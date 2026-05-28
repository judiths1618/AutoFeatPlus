"""
Transformer-based relation discovery — drop-in alternative to Valentine.

The key idea is that metadata.txt files (e.g. 6GDALI EUR) describe each column
in natural language ("time: timestamp of collection of metrics"). A
sentence-transformer embedding of "<col_name>: <description>" captures both
identifier-level and semantic similarity, which a name-only matcher misses.
Join-key candidates are then confirmed with a cheap value-overlap (Jaccard)
check, so we only emit edges where both schemas AND values agree.
"""

from __future__ import annotations

import os

# Force the HuggingFace `transformers` backend to PyTorch only. We use a PyTorch
# sentence-transformer; without this, transformers probes its TensorFlow backend
# and crashes with `ModuleNotFoundError: No module named 'tf_keras'` on Keras 3
# environments. Must be set before transformers/sentence_transformers import.
os.environ.setdefault("USE_TF", "0")

import glob
import itertools
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_discovery.config import CONNECTIONS, DATA_FOLDER
from feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables

logger = logging.getLogger(__name__)


# ─── Metadata parsing ────────────────────────────────────────────────────────
_FILES_LINE = re.compile(r"^\s*files?\s*:\s*(.+)$", re.IGNORECASE)
_COL_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$")
_TRIGGER_PHRASE = "information available"


def parse_metadata(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    """Return ``{filename: {column: description}}`` parsed from metadata.txt.

    The format we target (EUR 6GDALI) groups multiple files under one section
    and lists ``files: a.csv, b.csv`` or ``file: a.csv`` followed by a free-text
    block, then the literal phrase containing ``information available`` after
    which each line is ``<col>: <description>``.

    Columns not mentioned in the metadata still get an entry of the form
    ``"<col>: <col>"`` upstream (this function only emits what it parsed).
    """
    if not metadata_path.exists():
        logger.warning(f"metadata.txt not found at {metadata_path}; falling back to column names only")
        return {}

    text = metadata_path.read_text(encoding="utf-8")
    sections: List[Tuple[List[str], Dict[str, str]]] = []
    current_files: List[str] = []
    current_cols: Dict[str, str] = {}
    collecting_cols = False

    def flush():
        if current_files and current_cols:
            sections.append((list(current_files), dict(current_cols)))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        files_match = _FILES_LINE.match(line)
        if files_match:
            # Starting a new section: flush the previous one.
            flush()
            current_files = [f.strip() for f in files_match.group(1).split(",") if f.strip().endswith(".csv")]
            current_cols = {}
            collecting_cols = False
            continue

        if _TRIGGER_PHRASE in line.lower():
            collecting_cols = True
            continue

        if collecting_cols:
            col_match = _COL_LINE.match(line)
            if col_match:
                col_name, description = col_match.group(1), col_match.group(2).rstrip(". ")
                current_cols[col_name] = description

    flush()

    result: Dict[str, Dict[str, str]] = {}
    for files, cols in sections:
        for f in files:
            result.setdefault(f, {}).update(cols)
    return result


def _column_text(col_name: str, descriptions: Dict[str, str]) -> str:
    """Build the string we feed to the encoder for a single column."""
    desc = descriptions.get(col_name)
    if desc:
        return f"{col_name}: {desc}"
    # Fall back to a humanized version of the column name when no description.
    humanized = col_name.replace("_", " ").replace(".", " ")
    return f"{col_name}: {humanized}"


# ─── Embedding + matching ────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for transformer_discovery. "
            "Install with: pip install sentence-transformers"
        ) from e
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _embed(texts: List[str], model) -> np.ndarray:
    """Return L2-normalized embeddings so cosine = dot product."""
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.nan_to_num(np.asarray(embeddings, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)


def _value_jaccard(series_a: pd.Series, series_b: pd.Series, sample: int = 10_000) -> float:
    """Jaccard similarity over unique non-null values; cheap join-key sanity check."""
    a = series_a.dropna()
    b = series_b.dropna()
    if a.empty or b.empty:
        return 0.0
    if len(a) > sample:
        a = a.sample(sample, random_state=42)
    if len(b) > sample:
        b = b.sample(sample, random_state=42)
    set_a = set(a.unique().tolist())
    set_b = set(b.unique().tolist())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _match_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    descriptions_a: Dict[str, str],
    descriptions_b: Dict[str, str],
    model,
    schema_threshold: float,
    value_threshold: float,
) -> List[Tuple[str, str, float]]:
    """Return ``[(col_a, col_b, combined_score), ...]`` for one table pair."""
    cols_a = [c for c in df_a.columns if pd.api.types.is_hashable(c)]
    cols_b = [c for c in df_b.columns if pd.api.types.is_hashable(c)]
    if not cols_a or not cols_b:
        return []

    texts_a = [_column_text(c, descriptions_a) for c in cols_a]
    texts_b = [_column_text(c, descriptions_b) for c in cols_b]

    emb_a = _embed(texts_a, model)
    emb_b = _embed(texts_b, model)
    sim = emb_a @ emb_b.T  # cosine because both sides are normalized

    edges: List[Tuple[str, str, float]] = []
    for i, col_a in enumerate(cols_a):
        for j, col_b in enumerate(cols_b):
            schema_sim = float(sim[i, j])
            if schema_sim < schema_threshold:
                continue

            # Schema match but values can't be compared (different dtypes) →
            # keep as a low-confidence candidate, ranked below value-confirmed
            # matches by halving the score.
            if df_a[col_a].dtype != df_b[col_b].dtype:
                edges.append((col_a, col_b, schema_sim * 0.5))
                continue

            value_sim = _value_jaccard(df_a[col_a], df_b[col_b])
            # Filter: skip edges with neither strong value overlap nor very
            # strong schema match.
            if value_sim < value_threshold and schema_sim < 0.92:
                continue

            # Average of schema and value similarity. Schema-only matches
            # (value_sim == 0) are kept as candidates but capped at 0.5 *
            # schema_sim so they can never outrank a value-confirmed key.
            # This was the bug: previously `value_sim == 0` returned the full
            # schema_sim, so continuous-metric columns with identical schemas
            # (e.g. cpu_usage across tables, where instantaneous floats don't
            # overlap) tied with — and beat — true join keys that had partial
            # value overlap.
            if value_sim > 0:
                combined = 0.5 * schema_sim + 0.5 * value_sim
            else:
                combined = 0.5 * schema_sim
            edges.append((col_a, col_b, combined))
    return edges


# ─── Orchestrators (mirror the Valentine API) ────────────────────────────────
def profile_transformer_logic(
    files: List[str],
    metadata_path: Optional[Path],
    schema_threshold: float = 0.6,
    value_threshold: float = 0.2,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    write_connections_csv: Optional[Path] = None,
) -> List[dict]:
    """Run transformer discovery over ``files``. Edges are written to Neo4j and
    optionally also dumped to ``write_connections_csv`` for inspection.

    Returns the list of edges discovered.
    """
    descriptions = parse_metadata(metadata_path) if metadata_path else {}
    model = _get_model(model_name)
    discovered: List[dict] = []

    pairs = list(itertools.combinations(files, r=2))
    for tab_a, tab_b in tqdm(pairs, desc="transformer discovery"):
        a_path = tab_a.partition(f"{DATA_FOLDER}/")[2]
        b_path = tab_b.partition(f"{DATA_FOLDER}/")[2]
        a_name = Path(a_path).name
        b_name = Path(b_path).name

        try:
            df_a = pd.read_csv(tab_a, encoding="utf8")
            df_b = pd.read_csv(tab_b, encoding="utf8")
        except Exception as e:
            logger.warning(f"Skipping {a_name} ↔ {b_name}: {e}")
            continue

        edges = _match_pair(
            df_a, df_b,
            descriptions.get(a_name, {}), descriptions.get(b_name, {}),
            model,
            schema_threshold=schema_threshold,
            value_threshold=value_threshold,
        )

        for col_a, col_b, score in edges:
            logger.info(f"{a_name}.{col_a} ↔ {b_name}.{col_b}  (score={score:.3f})")
            merge_nodes_relation_tables(
                a_table_name=a_name, b_table_name=b_name,
                a_table_path=a_path, b_table_path=b_path,
                a_col=col_a, b_col=col_b, weight=score,
            )
            merge_nodes_relation_tables(
                a_table_name=b_name, b_table_name=a_name,
                a_table_path=b_path, b_table_path=a_path,
                a_col=col_b, b_col=col_a, weight=score,
            )
            discovered.append({
                "fk_table": a_name, "fk_column": col_a,
                "pk_table": b_name, "pk_column": col_b,
                "score": score,
            })

    if write_connections_csv is not None and discovered:
        write_connections_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(discovered).to_csv(write_connections_csv, index=False)
        logger.info(f"Wrote {len(discovered)} edges → {write_connections_csv}")

    return discovered


def profile_transformer_dataset(
    dataset_name: str,
    schema_threshold: float = 0.6,
    value_threshold: float = 0.2,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    metadata_path: Optional[Path] = None,
    write_connections_csv: Optional[Path] = None,
) -> List[dict]:
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]
    if metadata_path is None:
        # Default: look one directory above the dataset folder, then inside it.
        candidates = [
            DATA_FOLDER / dataset_name / "metadata.txt",
            (DATA_FOLDER / dataset_name).parent / "metadata.txt",
            DATA_FOLDER / "metadata.txt",
        ]
        metadata_path = next((p for p in candidates if p.exists()), None)

    return profile_transformer_logic(
        files=files,
        metadata_path=metadata_path,
        schema_threshold=schema_threshold,
        value_threshold=value_threshold,
        model_name=model_name,
        write_connections_csv=write_connections_csv,
    )


def profile_transformer_all(
    schema_threshold: float = 0.6,
    value_threshold: float = 0.2,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    metadata_path: Optional[Path] = None,
    write_connections_csv: Optional[Path] = None,
) -> List[dict]:
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]
    if metadata_path is None:
        candidates = list(Path(DATA_FOLDER).rglob("metadata.txt"))
        metadata_path = candidates[0] if candidates else None
    return profile_transformer_logic(
        files=files,
        metadata_path=metadata_path,
        schema_threshold=schema_threshold,
        value_threshold=value_threshold,
        model_name=model_name,
        write_connections_csv=write_connections_csv,
    )
