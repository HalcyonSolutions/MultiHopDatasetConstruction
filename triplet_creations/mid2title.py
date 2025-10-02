# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:26:01 2024

@author: Eduin Hernandez
"""
import argparse

import pandas as pd
from utils.basic import load_triplets, load_pandas

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="")
    
    # Input
    parser.add_argument('--triplets-path', type=str, default='./data/link_prediction/FB15k/triplets.txt',
                        help='Path to the text file containing valid triplets of entities used for filtering.')
    parser.add_argument('--mapping-data-path', type=str, default='./data/mappings/mid2name.tsv',
                        help='Path to the TSV file containing the MID to Title Mapping')

    parser.add_argument('--node-data-path', type=str, default='./data/metadata/node_data_fb15k.csv',
                        help='Path to save the node data for the dataset.')
    parser.add_argument('--relation-data-path', type=str, default='./data/metadata/relation_data_fb15k.csv',
                        help='Path to save the node data for the dataset.')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    triplet_df = load_triplets(args.triplets_path)
    mapping_data_df = pd.read_csv(args.mapping_data_path, sep='\t', header=None, names=['MID', 'Title'])
    
    valid_nodes = set(triplet_df['head']) | set(triplet_df['tail'])
    valid_rels = set(triplet_df['relation'])
    
    mapping_data_df = mapping_data_df[mapping_data_df['MID'].isin(valid_nodes)].drop_duplicates(subset='MID')
                  
    unknown_nodes = valid_nodes - set(mapping_data_df['MID'].tolist())
    
    unknown_nodes_df = pd.DataFrame({
    'MID': list(unknown_nodes),
    'Title': '[unknown]'
    })
    
    # Merge the new DataFrame with the existing one
    node_data_df = pd.concat([mapping_data_df, unknown_nodes_df], ignore_index=True)

    node_data_df.to_csv(args.node_data_path, index=False)

    relation_map = {}
    for v0 in valid_rels:
        v = v0.split('.')
        rel = []
        for v1 in v: 
            split = v1.split('/')
            if len(split) > 1: rel.append(f'{split[1]}/{split[-1]}')
            else: rel.append(v1)
        relation_map[v0] = "; ".join(list(set(rel)))

    relation_df = pd.DataFrame(list(relation_map.items()), columns=['Relation', 'Title'])
    relation_df.to_csv(args.relation_data_path, index=False)
