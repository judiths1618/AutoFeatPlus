import glob
import itertools
from typing import List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from valentine import valentine_match
from valentine.algorithms import Coma
from valentine.algorithms import JaccardDistanceMatcher
from valentine.algorithms import Cupid
from valentine.algorithms import DistributionBased
from valentine.algorithms import SimilarityFlooding  

from feature_discovery.config import DATA_FOLDER, CONNECTIONS, PROFILE
from feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables
import datasketch
from helpers.buildProfile import buildingProfile



def profile_valentine_all(valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    print(f"Found {len(files)} files to profile with Valentine.")
    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_dataset(dataset_name: str, valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_logic(files: List[str], valentine_threshold: float = 0.55):
    def profile(table_pair):
        (tab1, tab2) = table_pair

        a_table_path = tab1.partition(f"{DATA_FOLDER}/")[2]
        b_table_path = tab2.partition(f"{DATA_FOLDER}/")[2]

        a_table_name = a_table_path.split("/")[-1]
        b_table_name = b_table_path.split("/")[-1]

        print(f"Processing the match between:\n\t{a_table_path}\n\t{b_table_path}")
        df1 = pd.read_csv(tab1, encoding="utf8")
        df2 = pd.read_csv(tab2, encoding="utf8")

        print(f"Table 1: {df1.shape}, Table 2: {df2.shape}")
        # matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        # Instantiate matcher and run it
        # matcher = Coma(use_instances = True, java_xmx = "4g") # use_instances=True enables instance-based matching
        matcher = Coma()  # COMA matcher
        # matcher = JaccardDistanceMatcher()  # Jaccard distance matcher
        # matcher = SimilarityFlooding()  # Similarity flooding matcher
        # matcher = Cupid()  # Cupid matcher
        # matcher = DistributionBased()  # Distribution-based matcher

        # print(matcher)
        matches = valentine_match(df1, df2, matcher)

        for item in matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                print(f"Similarity {similarity} between:\n\t{a_table_path} -- {col_from}\n\t{b_table_path} -- {col_to}")

                merge_nodes_relation_tables(a_table_name=a_table_name,
                                            b_table_name=b_table_name,
                                            a_table_path=a_table_path,
                                            b_table_path=b_table_path,
                                            a_col=col_from,
                                            b_col=col_to,
                                            weight=similarity)

                merge_nodes_relation_tables(a_table_name=b_table_name,
                                            b_table_name=a_table_name,
                                            a_table_path=b_table_path,
                                            b_table_path=a_table_path,
                                            a_col=col_to,
                                            b_col=col_from,
                                            weight=similarity)

    Parallel(n_jobs=-1)(delayed(profile)(table_pair) for table_pair in tqdm(itertools.combinations(files, r=2)))


# offline compute
def filterDLake():
    dLakePath = f"{PROFILE}/LSHPROFILES"
    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    dLakeFiles =  glob.glob(dLakePath)

    print("data files", files, flush=True)
    print("profiles", dLakeFiles, flush=True)

    filesToProcess = []
    for f in files:
        if f not in dLakeFiles:
            filesToProcess.append(f)
    if len(filesToProcess) > 0:
        return f
    
    return -1


def profileDataLakeLSH(numPerms = 128, threshold=0.5):
    """ 
    Building the profiles of
    """
    if filterDLake() == -1:
        return 0
    files = filterDLake()
    for f in files:
        buildingProfile(f, threshold=threshold, numPerms=numPerms)


def buildLSHDataLake(dLake="default", numPerms = 128, threshold=0.5):
    
    import os
    dLakePath = f"{PROFILE}/{dLake}"
    profiles = os.listdir(dLakePath)

        


def insertLSHProfile(tablePair):
    table1 = tablePair[0]
    table2 = tablePair[1]

    a_table_path = table1.partition(f"{DATA_FOLDER}/")[2]
    b_table_path = table2.partition(f"{DATA_FOLDER}/")[2]


    a_table_name = a_table_path.split("/")[-1]
    b_table_name = b_table_path.split("/")[-1]

    lsh1 = 

    


def insetrBaseTableLSHIndex(file, dlake="default"):
    """
    Input shoudl be the base table, or table to be augmented
    The datalake of choice

    return is inserting the basetable into related join graph
    """
    if dlake == 'default':
        dLakePath =  f"{PROFILE}/LSHPROFILES"
    
    baseDF = pd.read_csv()




    
def profileDataLakeEmbedding():
    pass