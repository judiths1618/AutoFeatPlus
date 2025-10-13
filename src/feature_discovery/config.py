import os
from pathlib import Path

ROOT_FOLDER = Path(
    os.getenv("TFD_ROOT_FOLDER", Path(os.path.abspath(__file__)).parent.parent.parent.resolve())
).resolve()

print(f"TFD_ROOT_FOLDER: {ROOT_FOLDER}")

CONNECTIONS = "connections.csv"

DATASET_TYPE = "benchmark"

DATA = "data"
DATA_FOLDER = ROOT_FOLDER / DATA / DATASET_TYPE
# RESULTS_FOLDER = ROOT_FOLDER / "results" / "revision-test"
# RESULTS_FOLDER = ROOT_FOLDER / "results" / "thesis"
RESULTS_FOLDER = ROOT_FOLDER / "results" / "reproduction2025"
AUTO_GLUON_FOLDER = ROOT_FOLDER / "AutogluonModels"

print(f"DATA_FOLDER: {DATA_FOLDER}")
print(f"RESULTS_FOLDER: {RESULTS_FOLDER}")  

### CREDENTIALS ###
# NEO4J_HOST = os.getenv("NEO4J_HOST", "bolt://localhost:7689")
NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j://localhost:7689")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
NEO4J_CREDENTIALS = (NEO4J_USER, NEO4J_PASS)

NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", DATASET_TYPE)
NEO4J_DATABASE = "lake"
