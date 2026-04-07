# Artifact Reproduction for AutoFeat: Transitive Feature Discovery over Join Paths
This repo is a reproduced version of the development and experimental codebase of [AutoFeat: Transitive Feature Discovery over Join Paths](ICDE_FeatureDiscovery.pdf)


[![Python 3.8+](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pip](https://img.shields.io/badge/pip-20.0.2-blue.svg)](https://pypi.org/project/pip/)
[![Neo4j Desktop](https://img.shields.io/badge/neo4jDesktop-1.4.10-blue.svg)](https://pypi.org/project/pip/)


# 1. Development 
The code is available for local development, or using Docker. 

## Local development

### Requirements
- Python 3.10.18
- Java (for data discovery only - [Valentine](https://github.com/delftdata/valentine))
- neo4j 5.28.1

### Python setup 

1. Create virtual environment using Conda

`conda create -n {env-name} python=3.10 `

2. Activate environment 

`conda activate {env-name}`

3. Install requirements 

`pip install -e .`

#### Fix libomp
LighGBM on AutoGluon [gives Segmentation Fault](https://github.com/autogluon/autogluon/issues/1442) or won't run unless you install the corret libomp 
as described [here](https://github.com/autogluon/autogluon/pull/1453/files). 
Steps: 
```
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew uninstall libomp
brew install libomp.rb
rm libomp.rb
```

### Neo4j Desktop setup
Working with neo4j is easier using neo4j desktop application. 
1. First, download [neo4j Desktop](https://neo4j.com/download/)
2. Open the app
   1. "Add" > "Local DBMS"
   ![neo4j-create-dbms.png](assets%2Fneo4j-create-dbms.png)
   2. Give a name to the DBMS, add a password, and choose Version 5.1.0. 
   ![neo4j-create-db.png](assets%2Fneo4j-create-db.png)
   3. Change the "password" in [config](src/feature_discovery/config.py)
`NEO4J_PASS = os.getenv("NEO4J_PASS", "password")`
   3. "Start" the DBMS
   ![neo4j-open-database.png](assets%2Fneo4j-open-database.png)
   4. Once it started, "Open"
   ![neo4j-browser-open.png](assets%2Fneo4j-browser-open.png)
   5. Now you can see the neo4j browser, where you can query the database or create new ones, as we will do in the next steps. 


## Docker
The Docker image already contains all the necesarry for development.

1. Open a terminal and go to the project root (where the docker-compose.yml is located). 
2. Build necessary Docker containers (Note: This step takes a while)
``` bash
   docker-compose up -d --build
```

# 2. Data setup
1. [Download](https://surfdrive.surf.nl/files/index.php/s/1t1MTW8s8cfTDwc) our experimental datasets and put them in [data/benchmark](data/benchmark).

2. [Download](https://zenodo.org/records/6907619) we consider published 6G datasets and put them in [data/6g_testbed_dataset/EUR/](data/6g_testbed_dataset/EUR/). Refer to [docs/6g_dataset_setup.md](docs/6g_dataset_setup.md) for guidance on creating the corresponding `datasets.csv`, configuring `DATASET_TYPE`, and running Valentine-based relationship discovery when the CSV files do not expose explicit primary/foreign keys.

To ingest the data in the local development, it is necessary to follow the steps from [Neo4j Desktop setup](#neo4j-desktop-setup) beforehand.

For Docker, Neo4j browser is available at [localhost:7474](localhost:7474). No user or password is required.



## Benchmark setting

1. Create database `benchmark` in neo4j.
   1. Local development - It is necessary to follow the steps from [Neo4j Desktop setup](#neo4j-desktop-setup) beforehand.
   2. Docker - Go to [localhost:7474](localhost:7474) to access neo4j browser.

Input in neo4j browser console: 
![neo4j-console.png](assets%2Fneo4j-console.png)

```
create database benchmark 
```
Wait 1 minute until the database becomes available.
```
:use benchmark
```
2. Ingest data

-  (Docker) Bash into container 
```bash
   docker exec -it feature-discovery-runner /bin/bash
```
-  (Local development) Open a terminal and go to the project root. 

- Ingest the data using the following command:

```bash
 feature-discovery-cli ingest-kfk-data
```


## Data Lake setting
1. Go to [config.py](src/feature_discovery/config.py) and set `NEO4J_DATABASE = 'lake'`
   2. If Docker is running, restart it. 
2. Create database `lake` in neo4j:
   1. Local development - It is necessary to follow the steps from [Neo4j Desktop setup](#neo4j-desktop-setup) beforehand.
   2. Docker - Go to [localhost:7474](localhost:7474) to access neo4j browser.
   The default port will be 7474 for neo4j, need to change this in the config.py file to the available port that neo4j will be open on.


Input in neo4j browser console: 
![neo4j-console.png](assets%2Fneo4j-console.png)

```
create database lake
```
Wait 1 minute until the database becomes available.
```
:use lake
```  
3. Ingest data - depending on how many cores you have, this step can take up to 1-2h.

-  (Docker) Bash into container 
```bash
   docker exec -it feature-discovery-runner /bin/bash
```
-  (Local development) Open a terminal and go to the project root. 

- Ingest the data using the following command:

```
feature-discovery-cli ingest-data --data-discovery-threshold=0.55 --discover-connections-data-lake
```

Change the discovery mechansim, and the memory parameter, ran into out of memory error so need to configure


# 3. Experiments

To run the experiments in Docker, first bash into the container:
```bash
docker exec -it feature-discovery-runner /bin/bash
```

All experiment commands accept the same core arguments:

- `--dataset-labels` (required) – one or more labels from `data/benchmark/datasets.csv` (or the dataset type you exported).
- `--results-file` (optional) – override the default CSV name written to [`results/`](results).

You can inspect additional flags with `feature-discovery-cli <command> --help`.

### Run the full AutoFeat suite (ARDA + base + AutoFeat)

```bash
feature-discovery-cli run-all --dataset-labels steel --results-file results/steel_run_all.csv
```

The command above sequentially executes the ARDA, base, and AutoFeat pipelines for the `steel` dataset and stores the merged
metrics in `results/steel_run_all.csv`.

### Run only the ARDA baselines

```bash
feature-discovery-cli run-arda --dataset-labels steel --results-file results/steel_arda.csv
```

Use this when you only need the ARDA benchmark without running any feature discovery steps.

### Run the base models

```bash
feature-discovery-cli run-base --dataset-labels steel --results-file results/steel_base.csv
```

This evaluates the baseline models on the raw table without any AutoFeat augmentations.

### Run AutoFeat with custom hyper-parameters

```bash
feature-discovery-cli run-autofeat \
  --dataset-labels steel \
  --results-file results/steel_autofeat.csv \
  --value-ratio 0.55 \
  --top-k 15 \
  --no-store-augmented-data
```

`--value-ratio` controls the maximum fraction of missing values tolerated when selecting features, and `--top-k` sets the
number of join-path features to keep per dataset. Adjust both depending on the sparsity of your data. Pass
`--no-store-augmented-data` when you want to evaluate AutoFeat without writing the intermediate augmented tables to disk.

## Datasets 

Main [source](https://huggingface.co/datasets/inria-soda/tabular-benchmark#source-data) for finding datasets.

| Dataset Label | Source | Processing strategy | 
| ------------- | ------ | --------- | 
| [jannis](data/jannis) | [openml](https://www.openml.org/search?type=data&sort=runs&id=45021&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [MiniBooNe](data/miniboone) | [openml](https://www.openml.org/search?type=data&sort=runs&id=44128&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [covertype](data/covertype) | [openml](https://www.openml.org/search?type=data&sort=runs&id=44159&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [EyeMovement](data/eyemove) | [openml](https://www.openml.org/search?type=data&sort=runs&id=44157&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) |
| [Bioresponse](data/bioresponse) | [openml](https://www.openml.org/search?type=data&sort=runs&id=45019&status=active) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) | 
| [school](data/school) | [ARDA Paper](http://www.vldb.org/pvldb/vol13/p1373-chepurko.pdf) | None | 
| [steel](data/steel) | [openml](https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=1504) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) |
| [credit](data/credit) | [openml](https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=31) | [short_reverse_correlation](https://github.com/kirilvasilev16/PythonTableDivider) |

## Plots

1. To recreate our plots, first download the results from [here](https://surfdrive.surf.nl/files/index.php/s/fIhQNikpFbemozv).
2. Add the results in the [results](results) folder.
 
2. Then, open the jupyter notebook: run in the root folder of the project: 
```bash
jupyter notebook
```

2. Open the file [Visualisations.ipynb](Visualisations.ipynb).
3. Run every cell. 

# 4. Empirical analysis of feature selection strategies

We conducted an empirical analysis of the most popular feature selection 
strategies based on relevance and redundancy.

These experiments are documented at: https://github.com/delftdata/bsc_research_project_q4_2023/tree/main/autofeat_experimental_analysis 

### Maintainer
This repository is created and maintained by [Andra Ionescu](https://andraionescu.github.io)
