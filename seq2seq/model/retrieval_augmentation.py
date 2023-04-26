import torch
import numpy as np
import faiss

def retrieval_augmentation( train_dataset,  index, ra_dataset, model, k=1):
    
    augemented_dataset = []
    
    for X in train_dataset:
    
        new_text_embedding = model.encode(X)

        # Find the most similar text in the dataset
        _, indices = index.search(new_text_embedding, k)
        similar_text = ra_dataset[indices[0][0]]
        #  find the corresponding SQL query for the most similar text
        query = ra_dataset['query'][indices[0][0]]
            

        # append the original text and its corresponding SQL query with separator token
        query = query.replace("\n", " ")
        query = query.replace("\t", " ")
        
        appended_text = X + "[SEP]" + query
        
        
        augemented_dataset.append(appended_text)
        
    return augemented_dataset


def create_index(ra_dataset, model):
    
    
    text_embeddings = model.encode(ra_dataset)
    index = faiss.IndexFlatL2(text_embeddings.shape[1])
    index.add(text_embeddings)
    return index


