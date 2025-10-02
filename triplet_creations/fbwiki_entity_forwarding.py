"""
Created on Mon Aug  5 15:35:29 2024

@author: Eduin Hernandez

Summary: Extracts the redirected entities from WikiData given the original entities set.
"""

import argparse

from utils.basic import load_to_set
from utils.wikidata_v2 import process_entity_forwarding, fetch_entity_forwarding

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extracts the redirected entities from WikiData given the original entities set.")
    
    # Input arguments
    parser.add_argument('--max-workers', type=int, default=10, 
                        help='Number of workers for scraping')
    parser.add_argument('--max-rows', type=int, default=None, 
                        help='Number of rows to read from the entity list. Default is None, which means all rows.')
    parser.add_argument('--verbose-error', action='store_true',
                        help='Whether to display the errors')
    
    # Output arguments
    parser.add_argument('--entity-list-path', type=str, nargs='+', default=['./data/vocabs/nodes_fb15k.txt'],
                        help='Path to the list of entities')
    parser.add_argument('--output-path', type=str, default='./data/temp/nodes_fb_wiki_15k_forwarding.csv',
                        help='Path to save the triplets')
    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    entity_list = list(load_to_set(args.entity_list_path))[:args.max_rows]

    df = process_entity_forwarding(
        entity_list=entity_list, 
        max_workers=args.max_workers, 
        verbose=args.verbose_error
    )

    df.to_csv(args.output_path, index=False)
    print("\nData processed and saved to", args.output_path)

    # #-----------------------------------------------------
    # # Sanity Check
    # client = Client()
    # res = fetch_entity_forwarding('Q77023152', client)
    # print(res)