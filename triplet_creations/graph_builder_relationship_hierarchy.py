# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:48:29 2024

@author: Eduin Hernandez

Summary: Makes a connection to Neo4j to upload relationship hierarchy and relationship information
"""

import argparse

from utils.fb_wiki_graph import RelHierGraph

from utils.basic import load_pandas, str2bool
from utils.configs import global_configs

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process relationship hierarchy data for Neo4j graph creation.")
    
    # Input arguments
    parser.add_argument('--complete-relationship-data-path', type=str, default='./data/relation_data_wiki.csv',
                        help='Path to the complete properties list CSV file.')
    parser.add_argument('--hierarchy-data-path', type=str, default='./data/relationships_hierarchy.txt',
                        help='Path to the relationships hierarchy (triplets) data file.')
    
    # Neo4j
    parser.add_argument('--config-path', type=str, default='./configs/configs.ini',
                        help='Path to the configuration file for Neo4j connection.')
    parser.add_argument('--database', type=str, default='relhierarchy',
                        help='Name of the Neo4j database to use.')
    
    parser.add_argument('--create-new-graph', type=str2bool, default='True',
                        help='Whether to clear the graph and add the nodes from scratch')
    
    parser.add_argument('--add-new-nodes', type=str2bool, default='True',
                        help='Whether to add only the nodes')
    parser.add_argument('--update-nodes-info', type=str2bool, default='True',
                        help='Whether to update the details of the nodes')
    parser.add_argument('--upload-triplets', type=str2bool, default='True',
                        help='Whether to upload the links between the nodes')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    'Load the Data'
    rel_data = load_pandas(args.complete_relationship_data_path)
    rel_nodes = rel_data["Property"].tolist()
    
    configs = global_configs('./configs/configs.ini')
    neo4j_parameters = configs['Neo4j']
    
    g = RelHierGraph(neo4j_parameters['uri'], neo4j_parameters['user'], 
                      neo4j_parameters['password'], database = args.database)
    
    #--------------------------------------------------------------------------
    'Empties and Create Graph'
    if args.create_new_graph:
        g.create_graph(rel_nodes) # Avoid using if a graph already exists as this will erase it
    elif args.add_new_nodes:
        g.create_new_nodes(rel_nodes)
    
    #--------------------------------------------------------------------------
    'Update Node Information'
    if args.update_nodes_info:
        g.update_nodes_base_information(rel_data)

    #--------------------------------------------------------------------------
    'Create Hierarchy Triplets'
    if args.upload_triplets:
        g.create_link_between_nodes(rel_data, args.hierarchy_data_path)