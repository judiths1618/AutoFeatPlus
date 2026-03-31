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


from feature_discovery.config import DATA_FOLDER,DATA, CONNECTIONS, PROFILE
from feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables
from datasketch import MinHash
from feature_discovery.helpers.buildLSHProfile import buildingProfile, collectLshProfiles, repoChecker
import chromadb



def profile_valentine_all(valentine_threshold: float = 0.55):


    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]
    print("files for valentine", files)


    # print(f"Found {len(files)} files to profile with Valentine.", flush=True)
    profile_valentine_logic(files, valentine_threshold)

def profile_valentine_dataset(dataset_name: str, valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/**/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_logic(files: List[str], valentine_threshold: float = 0.55):
    def profile(table_pair):
        (tab1, tab2) = table_pair

        a_table_path = tab1.partition(f"{DATA_FOLDER}")[2].rstrip("/\\")
        b_table_path = tab2.partition(f"{DATA_FOLDER}")[2].rstrip("/\\")

        a_table_name = a_table_path.split("/")[-1]
        b_table_name = b_table_path.split("/")[-1]

        print(f"Processing the match between:\n\t{a_table_path}\n\t{b_table_path}")
        df1 = pd.read_csv(tab1, encoding="utf8")
        df2 = pd.read_csv(tab2, encoding="utf8")

        # print(f"Table 1: {df1.shape}, Table 2: {df2.shape}")
        # matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))

        # Instantiate matcher and run it
        # matcher = Coma(use_instances = True, java_xmx = "4g") # use_instances=True enables instance-based matching
        schema_matcher = Coma(java_xmx="256m")  # COMA matcher
        # instanance_matcher =  Coma(use_instances = True, java_xmx="256m") # use_instances=True enables instance-based matching

        instance_matcher = JaccardDistanceMatcher()  # Jaccard distance matcher
        # matcher = SimilarityFlooding()  # Similarity flooding matcher
        # matcher = Cupid()  # Cupid matcher
        # matcher = DistributionBased()  # Distribution-based matcher

        # print(matcher)
        

        #### Schema matching for unionable relations
        # Ran into memory error when computing instance-based matching on the whole dataset, so performing schema-based matching first to reduce the number of comparisons for instance-based matching
        schema_matches = valentine_match(df1, df2, schema_matcher)
        cols1 = set()
        cols2 = set()
        unionSchemas = set()
        for item in schema_matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                cols1.add(col_from)
                cols2.add(col_to)
                unionSchemas.add((col_from, col_to))
        cols1 = list(cols1)
        cols2 = list(cols2)        
        schemaMatchedDF1 = df1[cols1]
        schemaMatchedDF2 = df2[cols2]


        # instanance_matcher =  Coma(use_instances = True, java_xmx="256m") # use_instances=True enables instance-based matching

        # ##### perform instance check on unionable relations as joinable relations are a subset of unionable relations
        instance_matches = valentine_match(schemaMatchedDF1, schemaMatchedDF2, instance_matcher)

        joinableSchema = set()        
        for item in instance_matches.items():
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                joinableSchema.add((col_from, col_to))
                print(f"Similarity {similarity} between:\n\t{a_table_path} -- {col_from}\n\t{b_table_path} -- {col_to}")

                merge_nodes_relation_tables(a_table_name=a_table_name,
                                            b_table_name=b_table_name,
                                            a_table_path=a_table_path,
                                            b_table_path=b_table_path,
                                            a_col=col_from,
                                            b_col=col_to,
                                            weight=similarity,
                                            type="join")

                merge_nodes_relation_tables(a_table_name=b_table_name,
                                            b_table_name=a_table_name,
                                            a_table_path=b_table_path,
                                            b_table_path=a_table_path,
                                            a_col=col_to,
                                            b_col=col_from,
                                            weight=similarity,
                                            type="join")
                
        for pairs in joinableSchema:
            if pairs in unionSchemas:
                unionSchemas.remove(pairs)
        for pairs in unionSchemas:
            col_from, col_to = pairs
            print(f"Similarity {similarity} between:\n\t{a_table_path} -- {col_from}\n\t{b_table_path} -- {col_to}")

            merge_nodes_relation_tables(a_table_name=a_table_name,
                                        b_table_name=b_table_name,
                                        a_table_path=a_table_path,
                                        b_table_path=b_table_path,
                                        a_col=col_from,
                                        b_col=col_to,
                                        weight=similarity,
                                        type="union")

            merge_nodes_relation_tables(a_table_name=b_table_name,
                                        b_table_name=a_table_name,
                                        a_table_path=b_table_path,
                                        b_table_path=a_table_path,
                                        a_col=col_to,
                                        b_col=col_from,
                                        weight=similarity,
                                        type="union")   

                
        

    Parallel(n_jobs=-1)(delayed(profile)(table_pair) for table_pair in tqdm(itertools.combinations(files, r=2)))


# offline compute
def filterDLake(dLakePath):
    """
    Filter as per datalake:
    Input: dLake - the datalake to filter for

    Output: List of files to compute or -1 for nothing to compute
    """

    files = glob.glob(f"{DATA_FOLDER}/**/*.csv", recursive=True)
    dLakeFiles =  glob.glob(f"{dLakePath}/*.pkl", recursive=True)


    dLakeFiles = [f.partition(dLakePath)[-1].rstrip("/\\").lstrip("/\\").split(".pkl")[0] for f in dLakeFiles]
    # print("data files", files, flush=True)
    # print("profiles", dLakeFiles, flush=True)
    filesToProcess = []
    print("filtering datalake files ...", flush=True)
    for f in files:
        fileName = f.partition(str(DATA_FOLDER))[-1].rstrip("/\\").lstrip("/\\").split("\\")[-1].split(".csv")[0]
        # print("current file", fileName, flush=True)
        if fileName == "datasets" or fileName == "splitSummary":
            continue
        if fileName.split(".csv")[0] not in dLakeFiles:
            filesToProcess.append(f)
    if len(filesToProcess) > 0:
        return filesToProcess
        
    return -1


def profile_LSH_all(dLake = None, numPerms = 128, threshold=0.5):
    """ 
    Building all the minhash profiles of the datalake

    numPerms - number of permutations to be used in MinHash Construction
    threshold - similarity threshold for the LSH index

    """
    if dLake is None:
        dLakePath = f"{PROFILE}/LSHPROFILES/Global"
    else:
        dLakePath = f"{PROFILE}/LSHPROFILES/{dLake}"

    repoChecker(dLakePath)
    filterRes = filterDLake(dLakePath)

    # print("filterRes", filterRes)

    if filterRes == -1:
        return 0
    dLakeCollection = {}
    # construction lsh collection for future node additions

    # print("filtered datasets", filterRes)

    for f in tqdm(filterRes):
        minHashCollection = buildingProfile(f,dLake=dLake, threshold=threshold, numPerms=numPerms)
        if minHashCollection == {}:
            continue
        dLakeCollection[f] = minHashCollection
    
    collectLshProfiles()

    # construct the relationgraph from the hashes
    
    fileKeys = dLakeCollection.keys()

    ####################
    # need to parallelize this
    print("Initializing the relation graph with LSH profiles ...", flush=True)
    f
    for f in fileKeys:
        buildLSHDataLake(f, dLakeCollection[f])

        
def profileDataLakeLSH(dLake=None, numPerms = 128, threshold=0.5):
    """ 
    Building the profiles of
    """

    filterRes = filterDLake(dLake)
    if filterRes == -1:
        return 0

    dLakeCollection = {}
    # construction lsh collection for future node additions
    for f in filterRes:
        minHashCollection = buildingProfile(f, threshold=threshold, numPerms=numPerms)
        if minHashCollection == {}:
            continue
        dLakeCollection[f] = minHashCollection
    
    collectLshProfiles()

    # construct the relationgraph from the hashe
    fileKeys = dLakeCollection.keys()
    ####################
    # need to parallelize this
    for f in fileKeys:
        buildLSHDataLake(f, dLakeCollection[f])
               
    

def buildLSHDataLake(hashes, dLake=None,  threshold=0.5):
    
    import os
    import pickle

    if dLake is None:
        dLakeProfilePath = f"{PROFILE}/LSHProfiles/Global"
    else:
        dLakeProfilePath = f"{PROFILE}/LSHProfiles/{dLake}"

    dLakeFilePath = dLakeProfilePath + "/{}"

    profiles = os.listdir(dLakeProfilePath)
    print("profiles in the directory", profiles)
    if "globalLsh.pkl" in profiles:
        with open(f"{dLakeProfilePath}\globalLSH.pkl") as f2:
            globalLSH = pickle.load(f2)
    else:
        raise FileExistsError("Global LSH Index needs to be constructed")
    
    for c in hashes:
        tempMinHash = hashes[c]
        tempRes = globalLSH.query(tempMinHash)
        if len(tempRes) > 0:
            file2, col2 = c.split("_")
            for tr in tempRes:
                file1, col1 = tr.split("_")
                merge_nodes_relation_tables(a_table_name=file1,
                                        b_table_name=file2,
                                        a_table_path=dLakeFilePath.format(file1),
                                        b_table_path=dLakeFilePath.format(file2),
                                        a_col=col1,
                                        b_col=col2,
                                        weight=threshold)

                merge_nodes_relation_tables(a_table_name=file2,
                                        b_table_name=file1,
                                        a_table_path=dLakeFilePath.format(file2),
                                        b_table_path=dLakeFilePath.format(file1),
                                        a_col=col2,
                                        b_col=col1,
                                        weight=threshold)

                

def insertBaseTableLSHIndex(filepath, dLake=None, numPerms=128, threshold=0.5):
    """
    Input shoudl be the base table, or table to be augmented
    The datalake of choice

    return is inserting the basetable into related join graph
    """

    if dLake is None:
        dLakeProfilePath = f"{PROFILE}/LSHProfiles/Global"
    else:
        dLakeProfilePath = f"{PROFILE}/LSHProfiles/{dLake}"
    dLakeFilePath = dLakeProfilePath + "/{}"

    from pathlib import Path
    if Path(f"{filepath}").exists():
        baseDF = pd.read_csv(baseDF)
    else:
        raise FileNotFoundError("Base table does not exist, please check the table")

    string_col_names = list(baseDF.select_dtypes(include=["object", "string"]).columns)


    # check if global lsh index exists

    baseTableMinHashCollection = {}
    for col in string_col_names:
        tempMinHash = MinHash(threshold=threshold, num_perm=numPerms)
        baseTableMinHashCollection[f"{filepath}_{col}"] = tempMinHash

    import pickle
    with open(f"{dLakeProfilePath}/globalLsh.pkl", "rb") as f2:
        globalLSH = pickle.load(f2)
    

    for c in baseTableMinHashCollection:
        tempMinHash = baseTableMinHashCollection[c]
        tempRes = globalLSH.query(tempMinHash)
        if len(tempRes) > 0:
            file2, col2 = c.split("_")
            for tr in tempRes:
                file1, col1 = c.split("_")
                merge_nodes_relation_tables(a_table_name=file1,
                                        b_table_name=file2,
                                        a_table_path=dLakeFilePath.format(file1),
                                        b_table_path=dLakeFilePath.format(file2),
                                        a_col=col1,
                                        b_col=col2,
                                        weight=threshold)

                merge_nodes_relation_tables(a_table_name=file2,
                                        b_table_name=file1,
                                        a_table_path=dLakeFilePath.format(file2),
                                        b_table_path=dLakeFilePath.format(file1),
                                        a_col=col2,
                                        b_col=col1,
                                        weight=threshold)




    
def profileDataLakeEmbedding():

    

    pass