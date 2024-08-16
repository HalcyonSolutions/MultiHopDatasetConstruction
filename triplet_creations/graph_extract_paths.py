# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:10:04 2024

@author: Eduin Hernandez
"""

import argparse
import os

import random
import string
from itertools import zip_longest
import csv

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.fb_wiki_graph import FbWikiGraph, NodeRelationshipFilter
from utils.basic import load_pandas
from utils.configs import global_configs

#------------------------------------------------------------------------------
def str2bool(string):
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif string.lower() in ('none'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')      

def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Path Extraction using Neo4j.')
    
    'Node and Relationship info paths'
    parser.add_argument('--node-path', type=str, default='./data/category_mapping.csv', help='File path for Nodes data (csv).')
    parser.add_argument('--relation-path', type=str, default='./data/relation_data.csv', help='File path for Relationships data (csv).')
    parser.add_argument('--filter-path', type=str, default='./data/relationship_for_categories.csv', help='File path for relationship filtering according to the node data.')
    
    'Neo4j'
    parser.add_argument('--config-path', type=str, default='./configs/configs.ini', help='Configuration file for Neo4j access.')
    
    'Multi-hop Parameters'
    parser.add_argument('--min-hops', type=int, default=3, help='Minimum number of hops to consider in the path.')
    parser.add_argument('--max-hops', type=int, default=3, help='Maximum number of hops to consider in the path.')
    parser.add_argument('--max-attempts', type=int, default=1E9, help='Max number of attempts for extracting paths.')
    parser.add_argument('--num-workers', type=int, default=10, help='Number of workers to use for path extractions.')
    parser.add_argument('--use-filter', type=str2bool, default='False', help='Whether to filter out the path by using the relationships.')
    parser.add_argument('--use-pruning', type=str2bool, default='True', help='Whether to remove non-informative relationships.')
    parser.add_argument('--use-rand-path', type=str2bool, default='False', help='Whether to search and return paths randomly between two pairs. Warning! This could slow down path finding algorithms.')
    parser.add_argument('--path-per-pair', type=int, default=1, help='Maximum number of paths to generate per head and tail pairing.')
    parser.add_argument('--total-paths', type=int, default=2E4, help='Total of number of paths to create.')
    
    'Saving Parameters'
    parser.add_argument('--dataset-folder', type=str, default='./data/multi_hop/', help='Output directory of the path being generated')
    parser.add_argument('--dataset-prefix', type=str, default='temp', help='Output prefix of the file being generated. The full name will be {prefix}_hop_{suffix}.csv')
    parser.add_argument('--dataset-suffix', type=str, default='', help='Output suffix of the file being generated. The full name will be {prefix}_hop_{suffix}.csv')
    parser.add_argument('--reload-set', type=str2bool, default='False', help='Whether or not to continue using the previously created info. Use this if the simulation crashes')
    
    args = parser.parse_args()
    return args

def process_pair(g, x, y, args, nrfilter):
    """
    Function to process a pair (x, y) and return the paths found.
    """
    rels = None
    non_inform = []
    if args.use_filter: rels = nrfilter.nodes_rel_filters(x, y, remove_noninformative = args.use_pruning)
    if args.use_pruning and not(args.use_filter): non_inform = nrfilter.get_noninform_rels(neo4j = True)
    if args.use_filter: rels = nrfilter.nodes_rel_filters(x, y)
    
    paths = g.find_path(x, y, 
                        min_hops=args.min_hops,
                        max_hops=args.max_hops, 
                        relationship_types=rels,
                        noninformative_types=non_inform,
                        limit=args.path_per_pair,
                        rdf_only=True,
                        rand = args.use_rand_path
                        )
    return paths, x, y

if __name__ == '__main__':
    'Inputs'
    args = parse_args()
    
    #--------------------------------------------------------------------------
    'Input Check'
    
    #Node and Relationships
    assert os.path.isfile(args.node_path), 'Error! Node file does not exist!'
    assert os.path.isfile(args.relation_path), 'Error! Relationship file does not exist!'
    assert os.path.isfile(args.filter_path), 'Error! Filter file does not exist!'
    
    #Neo4j
    assert os.path.isfile(args.config_path), 'Error! Config file does not exist!'
    
    #Multihop
    assert not(args.min_hops > args.max_hops), 'Error! Max number of hops must be greater or equal to the minimum!'

    #--------------------------------------------------------------------------
    # Generate column names
    column_names = []
    for i in range(args.max_hops+1):
        column_names.append(string.ascii_uppercase[i])
        column_names.append(f'edge_{i+1}')
    column_names.pop() # remove last entry, which is a relationship
    
    #--------------------------------------------------------------------------
    'Load Pandas data'
    nodes = load_pandas(args.node_path)
    nodes_list = list(nodes['RDF'])
    assert args.total_paths < len(nodes_list) * (len(nodes_list) - 1) // 2, 'Number of pairs requested exceeds the number of possible unique pairs.'
    
    nrfilter = NodeRelationshipFilter(args.relation_path, args.filter_path, args.node_path)    

    #--------------------------------------------------------------------------
    'Load Neo4j Connection'
    configs = global_configs(args.config_path)
    neo4j_parameters = configs['Neo4j']
    
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'], neo4j_parameters['password'])
    
    # #--------------------------------------------------------------------------
    # 'Example of how to extract paths'
    # start_node = 'Q76' # Barack Obama
    # end_node = 'Q9682' # Elizabeth II
    # rels = None
    # non_inform = []
    # if args.use_filter: rels = nrfilter.nodes_rel_filters(start_node, end_node, remove_noninformative = args.use_pruning)
    # if args.use_pruning and not(args.use_filter): non_inform = nrfilter.get_noninform_rels(neo4j = True)
    
    # paths = g.find_path(start_node, end_node, 
    #                     min_hops=args.min_hops,
    #                     max_hops=args.max_hops, 
    #                     relationship_types=rels,
    #                     noninformative_types=non_inform,
    #                     limit=args.path_per_pair,
    #                     rdf_only=True,
    #                     rand = args.use_rand_path
    #                     )

    #--------------------------------------------------------------------------
    'Extracting Multihop Paths'    
    
    write_path = args.dataset_folder + args.dataset_prefix + '_hop'
    write_path += f"_{args.dataset_suffix}.csv" if args.dataset_suffix else '.csv'
    success_path = fail_path = args.dataset_folder + args.dataset_prefix 
    success_path += f"_{args.dataset_suffix}_success.txt" if args.dataset_suffix else '_success.txt'
    fail_path += f"_{args.dataset_suffix}_fails.txt" if args.dataset_suffix else '_fails.txt'

    pair_set = set()
    path_count = 0
    attempts = 0
    
    if args.reload_set: # Reuse previous information of pairs to continue
        assert os.path.isfile(write_path), 'Error! Dataset file has not been created yet!'
        assert os.path.isfile(success_path), 'Error! Success file has not been created yet!'
        assert os.path.isfile(fail_path), 'Error! Fail file has not been created yet!'

        f = open(success_path, "r")
        for i0 in f:
            pair_set.add(i0)
            attempts += 1
            path_count +=1
            
        f = open(fail_path, "r")
        for i0 in f:
            pair_set.add(i0)
            attempts += 1
    else: # Start fresh
        with open(write_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)  # Write the column names as the header row
        
        with open(fail_path, 'w') as f:
            pass
        
        with open(success_path, 'w') as f:
            pass
    
    # Pre-open the CSV file for writing
    with open(write_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Initialize tqdm progress bar for the main loop
        with tqdm(total=int(args.total_paths) - path_count, desc="Generating paths", unit="path") as pbar:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:  # Adjust max_workers based on your system
                while path_count < args.total_paths and attempts < args.max_attempts:
                    # Generate and submit tasks in a batch
                    future_to_pair = {}

                    while len(future_to_pair) < args.num_workers and path_count < args.total_paths:
                        attempts += 1
                    
                        x, y = random.sample(nodes_list, 2)  # prevents x == y
                        p = x + '_' + y
                        if p in pair_set: continue
                        pair_set.add(p)
    
                        # Submit the process_pair task to the thread pool
                        future = executor.submit(process_pair, g, x, y, args, nrfilter)
                        future_to_pair[future] = (x, y)
    
                    # Process the completed futures as they finish
                    for future in as_completed(future_to_pair):
                        paths, x, y = future.result()
                        p = x + '_' + y + '\n'
                        for path in paths:
                            tmp_path = [
                                item for pair in zip_longest(path[0], path[1])
                                for item in pair if item is not None
                            ]
                            writer.writerow(tmp_path)
                            # f.flush()
                            
                            path_count += 1
                            pbar.update(1)
                        
                        if paths:
                            with open(success_path, 'a') as f:
                                f.write(p + '')
                        else:
                            with open(fail_path, 'a') as f:
                                f.write(p)
                        # Break the loop if we have reached the required path count
                        if path_count >= args.total_paths: break