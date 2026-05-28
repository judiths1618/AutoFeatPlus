import os
from pathlib import Path

# Global random seed for every stochastic step in the pipeline (train/test
# splits, per-model RNGs, group sampling, feature-selection tie-breaks). Override
# with the AUTOFEAT_SEED env var or the auto_pipeline --seed flag.
SEED = int(os.getenv("AUTOFEAT_SEED", "42"))


ROOT_FOLDER = Path(
    os.getenv("TFD_ROOT_FOLDER", Path(os.path.abspath(__file__)).parent.parent.parent.resolve())
).resolve()


def rel(path) -> str:
    """Render *path* relative to the project root for display/logging.

    Keeps usernames and machine-specific prefixes (e.g. /Users/<name>/...) out of
    console output, logs and any committed result files. Paths outside the repo
    fall back to their leaf name rather than exposing the absolute path.
    """
    p = Path(path).resolve()
    if p == ROOT_FOLDER:
        return "."
    try:
        return str(p.relative_to(ROOT_FOLDER))
    except ValueError:
        return p.name


print(f"project root: {ROOT_FOLDER.name}")

CONNECTIONS = "connections.csv"


# Top-level data directory. Each subfolder (EUR/, KUL/, scenarioX_*/, ...) is a
# self-contained data-lake the augmentation pipeline can be pointed at via
# `auto_pipeline --data-dir`.
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", ROOT_FOLDER / "datasets")).resolve()
RESULTS_FOLDER = Path(os.getenv("RESULTS_FOLDER", ROOT_FOLDER / "results" / "6g_data")).resolve()
AUTO_GLUON_FOLDER = ROOT_FOLDER / "AutogluonModels"

PROFILE = "dataLakeProfiles/LSHProfiles"

print(f"DATA_FOLDER: {rel(DATA_FOLDER)}")
print(f"RESULTS_FOLDER: {rel(RESULTS_FOLDER)}")

### CREDENTIALS ###
# Use bolt:// for single-instance Neo4j; neo4j:// is for routing-capable clusters
# only and returns "Unable to retrieve routing information" against a standalone.
# The docker-compose service exposes bolt on 7687 with auth disabled.
NEO4J_HOST = os.getenv("NEO4J_HOST", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")
NEO4J_CREDENTIALS = (NEO4J_USER, NEO4J_PASS) if NEO4J_PASS else None

# Neo4j 5 forbids database names starting with a digit (so "6g_testbed_dataset"
# is rejected). Default to the built-in "neo4j" database; override with env var
# for a custom-named DB (Enterprise only).
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
