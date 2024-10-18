# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:35:29 2024

@author: Eduin Hernandez

Summary: Webscrapes WikiData to extract and create triplets

Stage 1
Previous Code: freebase_2_wikidata - part II (nodes_fb15k.txt)
Next Code: fbwiki_triplet_process (first-stage)

Make sure to remove error nodes that appear in failed_ent_log.txt from nodes_fb15k.txt

Stage 2
Previous Code: fbwiki_triplet_process (first-stage)
Next Code: fbwiki_triplet_process (second-stage)

Use it with the nodes missing that should be outputed in the first stage of fbwiki_triplet_process
"""

import argparse

from utils.basic import str2bool
from utils.wikidata_v2 import process_entity_triplets


def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Creates a triplet for the given list of entities by scrapping Wikidata")
    
    # Input arguments
    parser.add_argument('--max-workers', type=int, default=10, 
                        help='Number of workers for scraping')
    
    parser.add_argument('--verbose-error', type=str2bool, default='True', 
                        help='Whether to display the errors')
    
    # Output arguments
    parser.add_argument('--entity-list-path', type=str, nargs='+', default=['./data/nodes_fb15k.txt'],
                        help='Path to the list of entities')
    parser.add_argument('--output-path', type=str, default='./data/triplet_creation_fb15k.txt',
                        help='Path to save the triplets')
    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()

    process_entity_triplets(args.entity_list_path, args.output_path,
                            max_workers=args.max_workers, verbose=args.verbose_error)