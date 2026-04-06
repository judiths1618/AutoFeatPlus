import chromadb
import numpy as np
import pandas as pd
import os
from feature_discovery.config import PROFILE
from sentence_transformers import SentenceTransformer


def buildingEmbeddingSchemaProfile(file, dLake="Global", model="all-MiniLM-L6-v2"):

    """
    Building the Vector database used for embedding the schemas of the tables

    input:
        file-file name
        dlake - dlake name
        threshold - similarity threshold
        numPerms - number of permutations in minhash construction
    """

    profilePath = f"{PROFILE}/embeddingProfiles/{dLake}"
    client = chromadb.PersistentClient(path=profilePath)

    # if exists get, else create3
    collection = client.get_or_create_collection(
        name=f"schema_{dLake}",
        metadata={"description": f"Collection for the schema of the data lake {dLake}"}
    )

    print(f"Starting to Profile the dataset {file}", flush=True)
    # only need to read the headers
    dataset = pd.read_csv(file, nrows=10)
    cols = list(dataset.columns)

    file_and_col = [f"{file}_{col}" for col in cols]

    #Using default embedding model
    # Need to create a parameter that will take in embedding model to chaneg the embedding model
    # This will require a 
    collection.add(
        ids=file_and_col,
        documents = cols,
    )



def buildingEmbeddingInstProfile(file, dLake="Global", model="all-MiniLM-L6-v2"):

    """
    Building the Vector database used for embedding the schemas of the tables

    input:
        file-file name
        dlake - dlake name
        threshold - similarity threshold
        numPerms - number of permutations in minhash construction
    """

    profilePath = f"{PROFILE}/embeddingProfiles/{dLake}"
    client = chromadb.PersistentClient(path="./chroma_data")

    # if exists get, else create3
    collection = client.get_or_create_collection(
        name=f"schema_inst_{dLake}",
        metadata={"description": f"Collection for data lake {dLake} constructed using column embeddings from transformer models"}
    )

    print(f"Starting to Profile the dataset {file}", flush=True)
    dataset = pd.read_csv(file)
    cols = []
    for c in list(dataset.columns):
        if dataset[c].dtype == 'string':
            cols.append(c)
    

    # # embedding the instance set into a single embedding
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # for c in cols:
    #     tempData = list(dataset[c].unique())
    #     tempEmbeddings = model.encode(tempData)
    #     finalEmbedding = tempEmbeddings.sum(axis=0)
    #     collection.add({
    #         id = 
    #     })
        
    # This will be used as the IDs in the embedding db
    file_and_col = [f"{file}_{col}" for col in cols]

    #Using default embedding model
    # Need to create a parameter that will take in embedding model to chaneg the embedding model
    # This will require a
    # 
    aggEmbeddings = []
    for col in cols:
        tempData = list(dataset[col].unique())
        aggEmbedding = aggEmbeddingFunction(tempData, model)
        aggEmbeddings.append(aggEmbedding)
    #  
    colEmbeddings = collection.add(
        ids=file_and_col,
        embeddings=aggEmbeddings,
    )

    
    def aggEmbeddingFunction(documents, model="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model)
        output = model.encode(documents)
        agg_output = np.sum(output, axis=0)
        norm = np.linalg.norm(agg_output)  
        return agg_output / norm