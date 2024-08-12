# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:37:39 2024

@author: Eduin Hernandez

Summary: Analysis the tails occurances for a specific relationship
"""
import numpy as np
import pandas as pd

from utils.basic import sort_qid_list
from utils.process_triplets import extract_tail_occurrences_by_relationship, create_relationship_presence_df

if __name__ == '__main__':
    triplet_file_path = './data/modified_triplet.txt'
    filtered_file_path = './data/filtered_qid_triplet.txt'
    rel = 'P31'
    
    original = extract_tail_occurrences_by_relationship(triplet_file_path, rel)

    filtered = extract_tail_occurrences_by_relationship(triplet_file_path, rel)

    # Merge the DataFrames on 'tail' to compare the counts
    comparison_df = pd.merge(original, filtered, on='tail', how='outer', suffixes=('_original', '_filtered')).fillna(0)
    
    # Find rows where counts do not match
    mismatches = comparison_df[comparison_df['count_original'] != comparison_df['count_filtered']]
    
    # Display mismatches
    print("Mismatches between original and filtered data:")
    print(mismatches)