# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:37:39 2024

@author: Eduin Hernandez

Summary: Analysis the tails occurances for a specific relationship
"""
import numpy as np
import pandas as pd

from utils.basic import sort_qid_list
from utils.process_triplets import extract_tail_occurrences_by_relationship, create_relationship_presence_df

if __name__ == '__main__':
    'Input'
    triplet_file_path = './data/modified_triplet.txt'
    nodes_file_path = './data/modified_fbwiki_nodes.txt'
    node_data_path = './data/rdf_data.csv'
    instance_of = 'P31'
    
    'Output'
    output_path = './data/categories_list.csv'
    output_path2 = './data/category_mapping.csv'
    
    # Load the triplets and nodes
    triplet_df = pd.read_csv(triplet_file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    nodes = np.loadtxt(nodes_file_path, dtype=str).tolist()
    
    data_df = pd.read_csv(node_data_path)
    #--------------------------------------------------------------------------
    
    triplet_occurances = extract_tail_occurrences_by_relationship(triplet_file_path, instance_of)

    # Create the relationship presence DataFrame
    relationship_presence_df = create_relationship_presence_df(triplet_df, nodes, instance_of)
    relationship_presence_df.rename(columns={'head': 'RDF'}, inplace=True)
    relationship_presence_df.rename(columns={'has_relationship': 'has_category'}, inplace=True)
    relationship_presence_df.rename(columns={'tail': 'Category'}, inplace=True)
    
    no_instances = relationship_presence_df[(relationship_presence_df['has_relationship'] == False)]['head'].tolist()

    no_categories = sort_qid_list(set(no_instances) - set(triplet_occurances['tail'].tolist()))
    instance_categories    = sort_qid_list(triplet_occurances['tail'].tolist())
    
    full_categories = sort_qid_list(set(no_categories) | set(instance_categories))
    #---------------------------------------------------------------------------
    # Merge DataFrames on 'RDF' column
    assert data_df['RDF'].equals(relationship_presence_df['RDF']), "The 'RDF' columns do not match exactly"
    category_mapping = pd.merge(data_df, relationship_presence_df, on='RDF')
    category_mapping.to_csv(output_path2, header=True, index=False)
    
    categories_list = data_df[data_df['RDF'].isin(full_categories)]
    categories_list.to_csv(output_path, header=True, index=False)
    