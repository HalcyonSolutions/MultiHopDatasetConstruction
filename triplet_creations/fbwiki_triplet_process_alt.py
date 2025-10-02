# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:13:23 2024

@author: Eduin Hernandez

Summary: 
This script is an altertive to fbwiki_triplet_process, as it assumes a single stage for the processing due
already having the entity nodes as the heads, any node not present as head gets thrown away.
From original: It processes and refines triplet datasets derived from WikiData. It performs several key tasks aimed at 
cleaning and optimizing the triplet data for further analysis or use in knowledge graphs. 

Key functionalities:
1. **Node Pruning:** Filters out nodes (entities) that are irrelevant.
2. **Extraction of Entity and Relationship Sets:** Extracts sets of unique entities and relationships for further use, and saves the final processed triplet datasets.
3. **Statistics Reporting:** Outputs statistics on the number of triplets, entities, relationships, and missing entities.


Previous Codes: [fbwiki_triplet_creation, jeopardy_2_wikidata, fbwiki_relation_list_retrieval]
Next Code: [fbwiki_triplet_split, fbwiki_entity_data_retrival]

"""

import argparse

from utils.process_triplets import filter_triplets_by_entities, correct_forwarding
from utils.process_triplets import extract_triplet_sets
from utils.basic import load_triplets

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and refine triplet datasets from WikiData")
    
    # Input arguments
    parser.add_argument('--primary-triplet-path', type=str, nargs='+',  default=['./data/temp/triplet_creation_fb_wiki_all.txt'],
                            help='Paths to the primary triplet dataset(s) to process.')
    parser.add_argument('--entity-forwarding-path', type=str, default='./data/temp/nodes_fb_wiki_forwarding_v3.csv',
                            help='Path to the entity forwarding dataset.')																											   
    
    # Output arguments for triplet files
    parser.add_argument('--filtered-triplet-output', type=str, default='./data/temp/triplet_filt_fb_wiki_alt.txt',
                        help='Path to save the filtered triplet dataset (intermediate output).')
    
    # Output arguments for nodes and relationships
    parser.add_argument('--candidate-nodes-output', type=str, default='./data/vocabs/nodes_fb_wiki_alt.txt',
                        help='Path to save the candidate node set extracted from the triplets.')
    parser.add_argument('--candidate-relationships-output', type=str, default='./data/vocabs/relationship_fb_wiki_alt.txt',
                        help='Path to save the candidate relationship set extracted from the triplets.')
    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    
    #--------------------------------------------------------------------------
    # Step 0: If there is a redirection file, correct the triplet file to match the forwarding
    if args.entity_forwarding_path:
        args.primary_triplet_path = correct_forwarding(
            args.primary_triplet_path, 
            args.entity_forwarding_path, 
        )

    #--------------------------------------------------------------------------	
    triplets_df = load_triplets(args.primary_triplet_path) # Load the triplets to ensure the file exists and is readable									
    
    # Step 1: Collect entities and relationships for pruning and filtering
    entity_set = set(triplets_df['head'])
    
    # Step 2: Filter triplets based on the entity set and store the new triplets, contains duplicate removal
    filter_df = filter_triplets_by_entities(
        triplets_df, 
        entity_set, 
        args.filtered_triplet_output
    )
    
    #--------------------------------------------------------------------------
    # Step 4: Extract Information and Statistics
    
    extract_triplet_sets(
        triplet_processed=filter_df,
        triplet_original=None,
        nodes_candidates_path=args.candidate_nodes_output, 
        relationship_candidates_path=args.candidate_relationships_output, 
        nodes_missing_path=None
        )
    
    missing_nodes = entity_set - (set(filter_df['head']) | set(filter_df['tail']))
    print(f'Number of Missing Nodes: {len(missing_nodes)}')