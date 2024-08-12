# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:35:29 2024

@author: Eduin Hernandez

Summary: Webscrapes WikiData to extract and create triplets limited to the designited tails
"""
from utils.basic import load_json
# from utils.generate_triplets import generate_triplets, generate_triplets_threaded
from utils.generate_triplets import generate_triplets_valid_threaded



if __name__ == '__main__':
    'Input Files'
    input_set_path = './data/wiki_nodes.txt'
    valid_path = './data/fbwiki_nodes_mod.txt'
    property_path = './data/unique_properties_full.json'
    
    'Output Files'
    output_path = './data/qid_triplet_wiki.txt'
    
    #--------------------------------------------------------------------------
    property_map_rev = load_json(property_path)
    
    with open(output_path, 'w') as file:
        pass
    
    # generate_triplets(input_set_path, output_path, property_map_rev)
    generate_triplets_valid_threaded(input_set_path, output_path, property_map_rev, valid_path, max_workers=10)
