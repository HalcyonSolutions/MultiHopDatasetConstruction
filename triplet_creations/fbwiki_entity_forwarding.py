"""
Created on Mon Aug  5 15:35:29 2024

@author: Eduin Hernandez

Summary: Extracts the redirected entities from WikiData given the original entities set.
"""

import argparse

from utils.basic import str2bool
from utils.wikidata_v2 import process_entity_forwarding, fetch_entity_forwarding

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extracts the redirected entities from WikiData given the original entities set.")
    
    # Input arguments
    parser.add_argument('--max-workers', type=int, default=25, 
                        help='Number of workers for scraping')
    
    parser.add_argument('--verbose-error', type=str2bool, default='True', 
                        help='Whether to display the errors')
    
    # Output arguments
    parser.add_argument('--entity-list-path', type=str, nargs='+', default=['./data/nodes_fb_wiki_v3.txt'],
                        help='Path to the list of entities')
    parser.add_argument('--output-path', type=str, default='./data/nodes_fb_wiki_forwarding_v3.csv',
                        help='Path to save the triplets')
    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    process_entity_forwarding(args.entity_list_path, args.output_path,
                            max_workers=args.max_workers, verbose=args.verbose_error)

    # #-----------------------------------------------------
    # # Sanity Check
    # client = Client()
    # res = fetch_entity_forwarding('Q77023152', client)
    # print(res)