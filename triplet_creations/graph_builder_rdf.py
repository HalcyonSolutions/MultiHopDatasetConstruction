# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:30:28 2024

@author: Eduin Hernandez
"""

from typing import Dict, List
import json
import pandas as pd
import csv
import numpy as np

from utils.fb_wiki_graph import FbWikiGraph
from utils.basic import load_json
from utils.configs import global_configs

def load_rdf_valid(file_path: str) -> List[str]:
    return np.loadtxt(file_path, dtype=str).tolist()
    
# Function to load MDI to Title mapping from CSV
def load_rdf_info_mapping(file_path: str) -> Dict:
    mapping = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mapping[row['RDF']] = row
    return mapping
    
if __name__ == '__main__':
    # Constants for file paths
    RDF_NODES = './data/modified_fbwiki_nodes.txt'
    RDF_DATA_MAPPING = './data/category_mapping.csv'
    RELATION_MAPPING = './data/unique_properties_valid.json'
    FILE = './data/modified_triplet.txt'

    #--------------------------------------------------------------------------
    'Load the Data'
    rdf_data_map = load_rdf_info_mapping(RDF_DATA_MAPPING)
    relation_map = load_json(RELATION_MAPPING)
    nodes = load_rdf_valid(RDF_NODES)
    
    configs = global_configs('./configs/configs.ini')
    neo4j_parameters = configs['Neo4j']
    
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'], neo4j_parameters['password'])
    
    #--------------------------------------------------------------------------
    'Empties and Create Graph'
    # g.create_graph(rdf_valid) # Avoid using if a graph already exists as this will erase it

    g.create_new_nodes(nodes)
    
    #--------------------------------------------------------------------------
    'Update Node Information'
    g.update_nodes_base_information(rdf_data_map)
    g.update_node_category(rdf_data_map)
    
    #--------------------------------------------------------------------------
    'Connect Triplets'
    g.create_link_between_nodes(relation_map, FILE)
    
