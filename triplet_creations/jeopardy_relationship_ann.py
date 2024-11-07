# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:21:16 2024

@author: Eduin Hernandez

Summary: A sample code for mapping embeddings to the nearest relationship property/title
"""
import numpy as np

from utils.basic import load_embeddings
from utils.fb_wiki_ann import FbWikiANN
        
if __name__ == '__main__':
    
    ann = FbWikiANN(
            data_path = './data/relation_data_fj_wiki.csv',
            embedding_path = './data/relationship_embeddings_gpt_fj_wiki.csv', 
            exact_computation = True,
            nlist=130
            )
    
    topk = 10
    
    #--------------------------------------------------------------------------
    'ANN for multiple unknown embeddings at a time'
    embeddings_titles = load_embeddings('./data/relationship_embeddings_gpt_fj_wiki_titles.csv')
    embedding_vectors_titles = np.array(embeddings_titles['Embedding'].tolist())

    # Query the index to get top-K nearest neighbors for a matrix of embeddings
    distances, indices = ann.search(embedding_vectors_titles, topk)

    ground_truth = np.arange(ann.embedding_vectors.shape[0])

    hit1 = ann.calculate_hits_at_n(ground_truth, indices, 1)
    
    hit5 = ann.calculate_hits_at_n(ground_truth, indices, 5)
    
    hit10 = ann.calculate_hits_at_n(ground_truth, indices, 10)
    
    titles = ann.index2data(indices, 'Title')
    props = ann.index2data(indices, 'Property')
    
    print(f"Hit@1: {hit1:.4f}")
    print(f"Hit@5: {hit5:.4f}")
    print(f"Hit@10: {hit10:.4f}")
    
    mistakes_indices = ground_truth[ground_truth != indices[:, 0]]
    mistakes_truth = ann.data_df.iloc[mistakes_indices]['Title'].tolist()
    mistakes_predicted = np.array(titles)[mistakes_indices].squeeze().tolist()

    for pred, truth in zip(mistakes_predicted, mistakes_truth):
        print(f'Predicted: {pred:<40} \t\tGround: {truth}')
    # #--------------------------------------------------------------------------
    # 'ANN for single embedding at a time'
    # test_index = 20
    # # Query the index to get top-K nearest neighbors for a single
    # distance, index = ann.search(embedding_vectors_titles[test_index], topk)
    
    # title = ann.index2data(index, 'Title')
    # prop = ann.index2data(index, 'Property')
    # print('Ground Truth:', ann.data_df.iloc[test_index]['Title'])
    # print('Predicted Title:', title[0][0])
    # print('Predicted Property:', prop[0][0])
