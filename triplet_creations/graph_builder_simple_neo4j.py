# -*- coding: utf-8 -*-
"""
Created on 2025-04-26

@author: Eduin Hernandez

Summary: Makes a connection to Neo4j to upload triplet, entity, and relationship information
"""

import argparse
import debugpy

from utils.simple_graph import SimpleGraph
from utils.basic import load_pandas, load_to_set, load_triplets, str2bool
from utils.basic import overload_parse_defaults_with_yaml													 
from utils.configs import global_configs

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process entities data for Neo4j graph creation.")

    # Saved configs for posterity
    parser.add_argument('--saved-config', type=str, help='Path to a preconfigured save of arguments in a YAML config file')

    # Sample for FbWiki
    parser.add_argument('--entity-list-path', type=str, default='',
                        help='Path to the list of entities.')
    parser.add_argument('--triplets-data-path', type=str, nargs='+', default=['./data/link_prediction/KinshipHinton/triplets.txt'],
                        help='Path to the relationship between entities. Can be triplets or a list of training, testing, and validation triplets.')

    parser.add_argument('--max-workers', type=int, default=10, 
                        help='Number of workers to connect to Neo4j')
    parser.add_argument('--batch-size', type=int, default=20, 
                        help='Batch size (rows of data) to use per worker')
    
    # Neo4j
    parser.add_argument('--config-path', type=str, default='./configs/config_neo4j.ini',
                        help='Path to the configuration file for Neo4j connection.')
    parser.add_argument('--database', type=str, default='kinshiphinton',
                        help='Name of the Neo4j database to use.')
    
    parser.add_argument('--create-new-graph', type=str2bool,
                        help='Whether to clear the graph and add the nodes from scratch')
    parser.add_argument('--add-new-nodes', type=str2bool,
                        help='Whether to add only the nodes')
    parser.add_argument('--upload-triplets', type=str2bool,
                        help='Whether to upload the links between the nodes')

    parser.add_argument('--debug', '-d', action='store_true')

    args = parser.parse_args()

    if args.saved_config:
        print(
            f"\033[1;32mConfiguration loaded from {args.saved_config}."
            "\nThe config will override any CLI arguments.\033[0m"
        )
        args = overload_parse_defaults_with_yaml(args.saved_config, args)
        # Show me dump for sanity check
    else:
        print("\033[1;32mUsing default configuration\033[0m")
    
    return args
    
if __name__ == '__main__':
    
    args = parse_args()

    if args.debug:
        print("Waiting for debugger to attach...")
        debugpy.listen(("0.0.0.0", 42023))
        debugpy.wait_for_client()
        print("Debugger attached.")

    #--------------------------------------------------------------------------
    'Load the Data'

    if args.entity_list_path:
        nodes = list(load_to_set(args.entity_list_path))
    else:
        triplet_df = load_triplets(args.triplets_data_path)
        nodes = list(set(triplet_df['head']) | set(triplet_df['tail']))

    configs = global_configs(args.config_path)
    neo4j_parameters = configs['Neo4j']

    #--------------------------------------------------------------------------
    'Create Connection with Neo4j'
    g = SimpleGraph(neo4j_parameters['uri'], neo4j_parameters['user'],
                      neo4j_parameters['password'], database = args.database)
    

    #--------------------------------------------------------------------------
    'Empties and Create Graph'
    print(f"Value of create_new_graph: {args.create_new_graph}")
    if args.create_new_graph:
        g.create_graph(nodes) # Avoid using if a graph already exists as this will erase it
    elif args.add_new_nodes:
        g.create_new_nodes(nodes)
        
    #--------------------------------------------------------------------------
    'Connect Triplets'
    if args.upload_triplets:
        g.create_link_between_nodes(
            args.triplets_data_path,
            max_workers=args.max_workers,
            batch_size=args.batch_size)
