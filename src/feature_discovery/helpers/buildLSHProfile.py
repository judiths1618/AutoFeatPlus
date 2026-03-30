from datasketch import MinHash, MinHashLSH
import pandas as pd
import pickle
from config import PROFILE

def buildingProfile(file, dLake="default", threshold=0.5, numPerms=128):

    """
    Building the Lsh profiles of a file in the datalake
    input:
        file-file name
        dlake - dlake name
        threshold - similarity threshold
        numPerms - number of permutations in minhash construction
    """

    dataset = pd.read_csv(file)
    cols = []
    for c in list(dataset.columns):
        if dataset[c].dtype == 'string':
            cols.append(c)

    lshIndex = MinHashLSH(threshold=threshold, numPerms=numPerms)
    minHashcollection = {}
    for c in cols:
        tempMinHash = MinHash(num_perm=numPerms)
        data = dataset[c].unique()
        for d in data:
            tempMinHash.update(d.encode('utf8'))
        minHashcollection[c] = tempMinHash        
        lshIndex.insert(f"{file.encode("utf8")}_{c.encode("utf8")}", tempMinHash)


    with open(f"{PROFILE}\LSHProfiles\{dLake}\{file}.pkl", "wb") as f:
        pickle.dump(lshIndex, f)
    
    print(f"Build LSH index for relevant cols of {file}")

    return minHashcollection


def collectLshProfiles(dLake="default", threshold=0.5, numPerms=128):
    """
    Merge all the individual LSHs and store in the proflies directory
    """

    import os
    dPath = f"{PROFILE}\LSHProfiles\{dLake}"

    globalLsh = MinHashLSH(threshold=threshold, num_perm=numPerms)

    files = os.listdir(dPath)

    # merge all of the LSH of files in datalake to produce a global index for the data lake
    for f in files:
        with open(f"{dPath}\{f}.pkl", "rb") as f1:
            tempLsh = pickle.load(f1)
    
        globalLsh = globalLsh.merge(tempLsh)

    
    with open(f"{dPath}\globalLSH.pkl", "wb") as f2:
        pickle.dump(globalLsh, f2)


def repoChecker(dirPath):
    from pathlib import Path
    path = Path(dirPath)
    path.mkdir(parents=True, exist_ok=True)


