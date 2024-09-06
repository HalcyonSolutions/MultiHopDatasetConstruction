# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:13:23 2024

@author: Eduin Hernandez

Summary: Processes the triplets such that it prunes nodes that barely appear, removes relationships that go unused, 
            eliminates duplicates triplets, and fixes inverse relationship.
"""
import matplotlib.pyplot as plt

from utils.process_triplets import filter_head_tail, filter_triplets
from utils.process_triplets import count_head_tail, count_relationships
from utils.process_triplets import process_inverse_triplets, extract_and_save_rdf_titles

if __name__ == '__main__':
    # Input Files
    triplet_file_path = './data/qid_triplet_candidates.txt'
    relationship_mapping = './data/unique_properties_valid.csv'
    
    # Output Files
    filtered_triplet_file_path = './data/filtered_qid_triplet.txt'
    modified_triplet_file_path = './data/modified_triplet.txt'
    output_properties_json_path = './data/unique_properties_valid.json'
    # output_rdf_csv_path = './data/rdf_degree.csv'
    fbwiki_file_path = './data/modified_fbwiki_nodes.txt'
    
    #--------------------------------------------------------------------------
    # Step 1: Count head/tail occurrences and get the entity list
    entity_list = filter_head_tail(triplet_file_path, pruning_num=5)
    
    # Step 2: Filter triplets based on the entity list and store the new triplets
    filter_triplets(triplet_file_path, entity_list, filtered_triplet_file_path)
    
    # Step 3: Count the number of times each relationship and entity is used in the new triplets
    merged_counts, _, _ = count_head_tail(filtered_triplet_file_path)
    relationship_counts = count_relationships(filtered_triplet_file_path)
    
    #--------------------------------------------------------------------------
    # Step 4: Modify triplets based on invert and switch conditions
    process_inverse_triplets(filtered_triplet_file_path, relationship_mapping, modified_triplet_file_path)
    
    # Step 5: Count the number of times each relationship and entity is used in the new triplets
    merged_counts2, _, _ = count_head_tail(modified_triplet_file_path)
    relationship_counts2 = count_relationships(modified_triplet_file_path)
    
    extract_and_save_rdf_titles(relationship_mapping, relationship_counts2, output_properties_json_path)
    
    #--------------------------------------------------------------------------
    plt.figure()
    plt.title('Frequency of the Relationships')
    plt.plot(relationship_counts['count'].values, label='pre-processing')
    plt.plot(relationship_counts2['count'].values, label='post-processing')
    plt.ylabel('Frequency')
    plt.xlabel('Rank')
    plt.grid()
    plt.legend()
    
    plt.figure()
    plt.title('Frequency of the Relationships (log)')
    plt.plot(relationship_counts['count'].values, label='pre-processing')
    plt.plot(relationship_counts2['count'].values, label='post-processing')
    plt.ylabel('Frequency')
    plt.xlabel('Rank')
    plt.grid()
    plt.legend()
    plt.yscale('log')