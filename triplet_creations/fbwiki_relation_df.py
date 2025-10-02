# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:59:59 2024

@author: Eduin Hernandez

Summary: Given the triplets and the full list of relationships in Wikidata, 
    it extracts the relationship data of the corresponding dataset 
"""

import argparse

from utils.basic import load_triplets, load_pandas

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Creates the Relationship Database for the Dataset")
    
    # Input
    parser.add_argument('--relationship-wiki-data-path', type=str, default='./data/metadata/relation_data_wiki.csv',
                        help='Path to the data of the relationship.')
    
    parser.add_argument('--triplets-data-path', type=str, default='./data/link_prediction/Fb-Wiki/triplets.txt',
                        help='Path to the relationship between entities.')
    
    # Output
    parser.add_argument('--relationship-output-data-path', type=str, default='./data/metadata/relation_data_fb_wiki.csv',
                        help='Path to the data of the relationship.')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    triplets = load_triplets(args.triplets_data_path)
    
    rels = set(triplets['relation'].tolist())
    
    relation_map = load_pandas(args.relationship_wiki_data_path)
    relation_data = relation_map[relation_map['Property'].isin(rels)]
    
    #--------------------------------------------------------------------------
    
    relation_data.to_csv(args.relationship_output_data_path, index=False)