# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:13:23 2024

@author: Eduin Hernandez

Summary: 
This script processes and refines triplet datasets derived from WikiData. It performs several key tasks aimed at 
cleaning and optimizing the triplet data for further analysis or use in knowledge graphs. 

Key functionalities:
1. **Node Pruning:** Filters out nodes (entities) that appear less frequently than a specified threshold, but preserves key relationships like categories and family ties.
2. **Inverse and Bidirectional Relationship Handling:** Removes inverse and bidirectional relationships, ensuring consistency and uniqueness in the dataset.
3. **Triplet Merging:** Merges missing triplets into the candidate set and removes duplicates or inverse relationships.
4. **Extraction of Entity and Relationship Sets:** Extracts sets of unique entities and relationships for further use, and saves the final processed triplet datasets.
5. **Statistics Reporting:** Outputs statistics on the number of triplets, entities, relationships, and missing entities.

This script supports a two-stage process:
- **First Stage:** Focuses on the initial processing of primary triplet datasets, including pruning and removing duplicates.
- **Second Stage:** Involves merging new triplets with previously processed triplets and finalizing the dataset.

Stage 1:
    Previous Codes: [fbwiki_triplet_creation, jeopardy_2_wikidata, fbwiki_relation_list_retrieval]
    Next Code: fbwiki_triplet_creation

After processing the first stage of triplets, make sure to create triplets for entity nodes set that only appear as tails to get the missing triplets.

Stage 2:
    Previous Code: fbwiki_triplet_creation
    Next Code: [fbwiki_triplet_split, fbwiki_entity_data_retrival]
"""

import argparse

from utils.basic import str2bool

from utils.process_triplets import filter_triplets_by_entities, clean_triplet_relations, process_and_merge_missing_triplets
from utils.process_triplets import collect_entities_via_pruning, collect_tails_given_relation
from utils.process_triplets import extract_triplet_sets

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and refine triplet datasets from WikiData")
    
    # Stage control
    parser.add_argument('--first-stage', type=str2bool, default='False',
                        help='Flag to specify if processing is in the first stage (True) or second stage (False).')
    
    # Input arguments
    parser.add_argument('--primary-triplet-path', type=str, nargs='+',  default=['./data/triplet_creation_fb15k.txt'],
                            help='Paths to the primary triplet dataset(s) to process.')
    # parser.add_argument('--primary-triplet-path', type=str, nargs='+',  default=['./data/triplet_creation_fb15k.txt', './data/triplet_creation_jeopardy.txt'],
    #                         help='Paths to the primary triplet dataset(s) to process.')
    parser.add_argument('--secondary-triplet-path', type=str, nargs='+',  default=['./data/triplet_missing_fb_wiki.txt'],
                            help='Paths to the secondary triplet dataset for merging with processed triplets.')

    parser.add_argument('--relationship-hierarchy-path', type=str, default='./data/relationships_hierarchy.txt',
                        help='Path to the file containing relationship hierarchies for processing.')
    
    # Filtering Argumnets
    parser.add_argument('--pruning-threshold', type=int, default=5, 
                        help='Minimum frequency of occurrence for nodes; nodes appearing fewer times will be pruned.')
    
    # Inverse and relationship processing
    parser.add_argument('--enable-inverse-removal', type=str2bool, default='True', 
                        help='Flag to remove inverse relationships during triplet processing (True/False).')
    parser.add_argument('--enable-bidirectional-removal', type=str2bool, default='True', 
                        help='Flag to remove bidirectional relationships (True/False).')
    parser.add_argument('--inverse-mapping-path', type=str, default='./data/relation_inverse_mapping_fb_wiki.txt',
                        help='Path to the file containing inverse relationship mappings.')
    parser.add_argument('--reverse-mapping-path', type=str, default='./data/relation_reverse_mapping_fb_wiki.txt',
                        help='Path to the file containing reverse relationship mappings.')
    
    # Output arguments for triplet files
    parser.add_argument('--filtered-triplet-output', type=str, default='./data/triplet_filt_fb_wiki.txt',
                        help='Path to save the filtered triplet dataset (intermediate output).')
    parser.add_argument('--processed-triplet-output', type=str, default='./data/triplet_candidates_fb_wiki.txt',
                        help='Path to save the processed triplet dataset (final output for stage 1).')
    parser.add_argument('--final-triplet-output', type=str, default='./data/triplets_fb_wiki.txt',
                        help='Path to save the fully processed and merged triplet dataset (final output for stage 2).')
    
    # Output arguments for nodes and relationships
    parser.add_argument('--candidate-nodes-output', type=str, default='./data/nodes_candidates_fb_wiki.txt',
                        help='Path to save the candidate node set extracted from the triplets.')
    parser.add_argument('--final-nodes-output', type=str, default='./data/nodes_fb_wiki.txt',
                        help='Path to save the final node set extracted from the processed triplets.')
    parser.add_argument('--candidate-relationships-output', type=str, default='./data/relationship_candidates_fb_wiki.txt',
                        help='Path to save the candidate relationship set extracted from the triplets.')
    parser.add_argument('--final-relationships-output', type=str, default='./data/relationship_fb_wiki.txt',
                        help='Path to save the final relationship set extracted from the processed triplets.')
    parser.add_argument('--missing-nodes-output', type=str, default='./data/nodes_missing_fb_wiki.txt',
                        help='Path to save the set of missing nodes identified during processing.')
    
    # Parse arguments
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    
    args = parse_args()
    
    if args.first_stage:
        'First Stage'
        #--------------------------------------------------------------------------
        # Step 1: Collect the entities set to keep by pruning low occurances, but maintaining crucial ones like categorical
        df = load_triplets(args.primary_triplet_path)
        entity_set = collect_entities_via_pruning(df, pruning_num=args.pruning_threshold)
        
        entity_set.update(collect_tails_given_relation(args.primary_triplet_path,
                                                        ['P31', 'P279', 'P361',
                                                          'P19', 'P20', 'P793', 'P157', 'P509',
                                                          'P22', 'P25', 'P26', 'P40', 'P1038', 'P3373',
                                                          'P1365', 'P1366'])) 
        # [instance of, subclass of, part of, 
        # place of birth, place of death, significant event, killed by, cause of death, 
        # father, mother, child, spouse, child, relative, sibling, 
        # replaces, replaced by]
        
        # Step 2: Filter triplets based on the entity set and store the new triplets
        filter_triplets_by_entities(
            args.primary_triplet_path, 
            entity_set, 
            args.filtered_triplet_output)
        
        #--------------------------------------------------------------------------
        # Step 3: Replace inverses and remove duplicates
        
        clean_triplet_relations(
            triplet_filtered_file_path=args.filtered_triplet_output,
            triplet_processed_file_path=args.processed_triplet_output,
            relationship_hierarchy_mapping=args.relationship_hierarchy_path,
            inverse_mapping_path=args.inverse_mapping_path,
            reverse_mapping_path=args.reverse_mapping_path,
            remove_inverse_relationships=args.enable_inverse_removal,
            remove_bidirectional_relationships=args.enable_bidirectional_removal
            )
        
        #--------------------------------------------------------------------------
        # Step 4: Extract Information and Statistics
        
        extract_triplet_sets(
            triplet_processed_file_path=args.processed_triplet_output,
            triplet_file_path=args.primary_triplet_path,
            nodes_candidates_path=args.candidate_nodes_output, 
            relationship_candidates_path=args.candidate_relationships_output, 
            nodes_missing_path=args.missing_nodes_output
            )
    else:
        'Second Stage'
        #----------------------------------------------------------------------
        # Step 1: Replace inverses and remove duplicates of new triplets, then filter them on the valid entities & rel sets 
        process_and_merge_missing_triplets( 
                missing_triplets_path = args.secondary_triplet_path,
                candidates_triplets_path = args.processed_triplet_output,
                triplets_output_path = args.final_triplet_output,
                nodes_candidates_path = args.candidate_nodes_output, 
                relationship_candidates_path = args.candidate_relationships_output,
                inverse_mapping_path = args.inverse_mapping_path,
                reverse_mapping_path = args.reverse_mapping_path,
                remove_inverse_relationships = args.enable_inverse_removal,
                remove_bidirectional_relationships = args.enable_bidirectional_removal
                )
        
        #--------------------------------------------------------------------------
        # Step 2: Extract Information and Statistics
        extract_triplet_sets(
            triplet_processed_file_path=args.final_triplet_output,
            nodes_candidates_path=args.final_nodes_output, 
            relationship_candidates_path=args.final_relationships_output, 
            )