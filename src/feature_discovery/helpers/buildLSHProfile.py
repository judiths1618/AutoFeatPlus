from fileinput import filename

from datasketch import MinHash, MinHashLSH
import pandas as pd
import pickle
from feature_discovery.config import PROFILE

def buildingProfile(file, dLake=None, threshold=0.5, numPerms=128):

    """
    Building the Lsh profiles of a file in the datalake
    input:
        file-file name
        dlake - dlake name
        threshold - similarity threshold
        numPerms - number of permutations in minhash construction
    """
 
    # print("Building LSh profile for file", file)
    if dLake is None:
        dLakePath = f"{PROFILE}/LSHPROFILES/Global"
    else:
        dLakePath = f"{PROFILE}/LSHPROFILES/{dLake}"

 
    fileName = file.split("\\")[-1].split(".csv")[0]
    dataset = pd.read_csv(file)
    cols = []
    for c in list(dataset.columns):
        if dataset[c].dtype == 'string':
            cols.append(c)
    file_val = file.encode("utf8")
    lshIndex = MinHashLSH(threshold=threshold, num_perm=numPerms)
    minHashcollection = {}

    # print("columns to be hashed", cols)
    for c in cols:
        tempMinHash = MinHash(num_perm=numPerms)
        data = dataset[c].unique()
        for d in data:
            tempMinHash.update(d.encode('utf8'))
        minHashcollection[c] = tempMinHash        
        c_val = c.encode("utf8")
        lshIndex.insert(f"{file_val}_{c_val}", tempMinHash)

    fileName = file.split("\\")[-1].split(".csv")[0]
    # print("filename ", fileName)

    with open(f"{dLakePath}\{fileName}.pkl", "wb") as f:
        pickle.dump(lshIndex, f)
    
    # print(f"Built LSH index for relevant cols of {file}")

    return minHashcollection


def collectLshProfiles(dLake=None, threshold=0.5, numPerms=128):
    """
    Merge all the individual LSHs and store in the proflies directory
    """

    import os
    if dLake is None:
        dLakePath = f"{PROFILE}/LSHPROFILES/Global"
    else:
        dLakePath = f"{PROFILE}/LSHPROFILES/{dLake}"

    globalLsh = MinHashLSH(threshold=threshold, num_perm=numPerms)

    print("LSH index for lake", globalLsh)

    files = os.listdir(dLakePath)


    # merge all of the LSH of files in datalake to produce a global index for the data lake
    for f in files:
        with open(f"{dLakePath}\{f}", "rb") as f1:
            tempLsh = pickle.load(f1)
        
        if globalLsh is None:
            globalLsh = tempLsh
        else:
            globalLsh = globalLsh.merge(tempLsh)
    
    with open(f"{dLakePath}\globalLSH.pkl", "wb") as f2:
        pickle.dump(globalLsh, f2)


def repoChecker(dirPath):
    from pathlib import Path
    path = Path(dirPath)
    path.mkdir(parents=True, exist_ok=True)


