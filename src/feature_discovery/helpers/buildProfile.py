from datasketch import MinHash, MinHashLSH
import pandas as pd
import pickle
from pathlib import Path
from feature_discovery.config import PROFILE

def buildingProfile(file, dlake="default", threshold=0.5, numPerms=128):

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

    lshIndex = MinHashLSH(threshold=threshold, num_perm=numPerms)
    minHashcollection = {}
    for c in cols:
        tempMinHash = MinHash(num_perm=numPerms)
        data = dataset[c].unique()
        for d in data:
            tempMinHash.update(d.encode('utf8'))
        minHashcollection[c] = tempMinHash        
        lshIndex.insert(f"{file}_{c}", tempMinHash)


    profile_path = Path(PROFILE) / dlake / f"{Path(file).name}.pkl"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("wb") as f:
        pickle.dump(lshIndex, f)
    
    print(f"Build LSH index for relevant cols of {file}")

    return minHashcollection


def collectLshProfiles(dLake="default", threshold=0.5, numPerms=128):
    """
    Merge all the individual LSHs
    """

    dPath = Path(PROFILE) / dLake

    globalLsh = MinHashLSH(threshold=threshold, num_perm=numPerms)

    # merge all of the LSH of files in datalake to produce a global index for the data lake
    for profile_file in dPath.glob("*.pkl"):
        if profile_file.name == "globalLSH.pkl":
            continue
        with profile_file.open("rb") as f1:
            tempLsh = pickle.load(f1)
    
        globalLsh = globalLsh.merge(tempLsh)

    
    with (dPath / "globalLSH.pkl").open("wb") as f2:
        pickle.dump(globalLsh, f2)
