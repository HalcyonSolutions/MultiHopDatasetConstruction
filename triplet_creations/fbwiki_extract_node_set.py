# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:33:22 2024

@author: Eduin Hernandez

Summary: Takes the original FB15k-237 nodes as QID and stores them as text file, 
         the extra nodes from WikiData in another file, and the combined node 
         set as a third file.
"""

import pandas as pd
import numpy as np

from utils.basic import load_to_set, sort_by_qid

if __name__ == '__main__':
    'Input Files'
    filtered_triplet_file_path = './data/filtered_qid_triplet.txt'
    
    'Output Files'
    fb_file_path = './data/fb15k_nodes.txt'
    wiki_file_path = './data/wiki_nodes.txt'
    fbwiki_file_path = './data/fbwiki_nodes.txt'
    
    df = pd.read_csv(filtered_triplet_file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    # Find unique heads
    unique_heads = df['head'].unique()
    
    # Find unique tails that are not found in heads
    unique_tails = df[~df['tail'].isin(df['head'])]['tail'].unique()
    
    # Convert to DataFrame for sorting
    unique_heads_df = pd.DataFrame(unique_heads, columns=['qid'])
    unique_tails_df = pd.DataFrame(unique_tails, columns=['qid'])
    unique_df = pd.concat([unique_heads_df, unique_tails_df])
    
    sorted_unique_heads_df  = sort_by_qid(unique_heads_df)
    sorted_unique_tails_df  = sort_by_qid(unique_tails_df)
    sorted_unique_df        = sort_by_qid(unique_df)
    
    np.savetxt(fb_file_path, sorted_unique_heads_df['qid'].values, fmt="%s")
    np.savetxt(wiki_file_path, sorted_unique_tails_df['qid'].values, fmt="%s")
    np.savetxt(fbwiki_file_path, sorted_unique_df['qid'].values, fmt="%s")
    
    print("Data sorted and saved!")
    
    # Reload the files and convert to sets
    fb_set = load_to_set(fb_file_path)
    wiki_set = load_to_set(wiki_file_path)
    fbwiki_set = load_to_set(fbwiki_file_path)
    