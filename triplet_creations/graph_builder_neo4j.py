# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:30:28 2024

@author: Eduin Hernandez

Summary: Makes a connection to Neo4j to upload triplet, entity, and relationship information
"""

import argparse

from utils.fb_wiki_graph import FbWikiGraph
from utils.basic import load_pandas, load_to_set, str2bool
from utils.configs import global_configs

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process entities data for Neo4j graph creation.")
    
    # Input arguments
    parser.add_argument('--entity-list-path', type=str, default='./data/nodes_fb_wiki.txt',
                        help='Path to the list of entities.')
    parser.add_argument('--entity-data-path', type=str, default='./data/node_data_fb_wiki.csv',
                        help='Path to the data of the entities.')
    parser.add_argument('--relationship-data-path', type=str, default='./data/relation_data_wiki.csv',
                        help='Path to the data of the relationship.')
    parser.add_argument('--triplets-data-path', type=str, default='./data/triplets_fb_wiki.txt',
                        help='Path to the relationship between entities.')
    
    parser.add_argument('--max-workers', type=int, default=10, 
                        help='Number of workers to connect to Neo4j')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Batch size (rows of data) to use per worker')
    
    # Neo4j
    parser.add_argument('--config-path', type=str, default='./configs/configs.ini',
                        help='Path to the configuration file for Neo4j connection.')
    parser.add_argument('--database', type=str, default='fbwiki',
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
    rdf_data_map = load_pandas(args.entity_data_path)
    relation_map = load_pandas(args.relationship_data_path)
    nodes = list(load_to_set(args.entity_list_path))
    
    configs = global_configs('./configs/configs.ini')
    neo4j_parameters = configs['Neo4j']
    
    #--------------------------------------------------------------------------
    'Process the Data'
    
    relation_map = relation_map.fillna('')
    rdf_data_map = rdf_data_map.fillna('')
    
    #--------------------------------------------------------------------------
    'Create Connection with Neo4j'
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'],
                    neo4j_parameters['password'], database = args.database)
    

    #--------------------------------------------------------------------------
    'Empties and Create Graph'
    if args.create_new_graph:
        g.create_graph(nodes) # Avoid using if a graph already exists as this will erase it
    elif args.add_new_nodes:
        g.create_new_nodes(nodes)
    
    #--------------------------------------------------------------------------
    'Update Node Information'
    if args.update_nodes_info:
        g.update_nodes_base_information(
            rdf_data_map,
            max_workers=args.max_workers, 
            batch_size=args.batch_size)
        
        # g.update_node_category(rdf_data_map)
    
    #--------------------------------------------------------------------------
    'Connect Triplets'
    if args.upload_triplets:
        g.create_link_between_nodes(
            relation_map,
            args.triplets_data_path,
            max_workers=args.max_workers,
            batch_size=args.batch_size)
