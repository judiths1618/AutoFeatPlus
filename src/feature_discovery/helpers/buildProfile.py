from datasketch import MinHash, MinHashLSH
import pandas as pd
import pickle
from config import PROFILE

def buildingProfile(file, dlake="default", threshold=0.5, numPerms=128):
    dataset = pd.read_csv(file)
    cols = []
    for c in list(dataset.columns):
        if dataset[c].dtype == 'string':
            cols.append(c)

    lshIndex = MinHashLSH(threshold=threshold, numPerms=numPerms)
    for c in cols:
        tempMinHash = MinHash(num_perm=numPerms)
        data = dataset[c].unique()
        for d in data:
            tempMinHash.update(d.encode('utf8'))
        
        lshIndex.insert(c.encode("utf8"), tempMinHash)


    with open(f"{PROFILE}\{dlake}\{file}.pkl", "wb") as f:
        pickle.dump(lshIndex, f)
    
    print(f"Build LSH index for relevant cols of {file}")