# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:30:25 2024

@author: Eduin Hernandez

Summary:
This script processes a dataset of relationships to either estimate the cost of embedding calculations or extract embeddings
for each relationship title and alias. It allows the user to specify paths for input and output data, as well as options 
for model selection, encoder type, and whether to calculate pricing or extract embeddings.

Usage:
    The script can be run from the command line with options to specify input paths, model settings, pricing calculation, 
    and embedding extraction.
"""
import argparse

from utils.basic import load_pandas, str2bool
from utils.openai_api import OpenAIHandler, pricing_embeddings

import numpy as np
from tqdm import tqdm

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process relationship data to estimate embedding costs or extract embeddings.")
    
    # Input
    parser.add_argument('--relation-data-path', type=str, default='./data/relation_data_fj_wiki.csv',
                        help='Path to the CSV file containing relationship data to be processed.')

    parser.add_argument('--model', type=str, default='text-embedding-3-small',
                        help='Model name to be used for embedding calculations (e.g., "text-embedding-3-small"). Must be a key in pricing_embeddings.')
    parser.add_argument('--encoder', type=str, default='cl100k_base',
                        help='Encoding name used by the model to tokenize text for embeddings.')

    parser.add_argument('--include-alias', type=str2bool, default='False',
                        help='Flag to use alias in the embedding')
    parser.add_argument('--calculate-pricing', type=str2bool, default='True',
                        help='Flag to calculate and display estimated embedding costs. Expects "True" or "False".')
    parser.add_argument('--extract-embedding', type=str2bool, default='False',
                        help='Flag to enable extraction of embeddings for each title and alias. Expects "True" or "False".')
    
    # Output
    parser.add_argument('--embedding-output-path', type=str, default='./data/embeddings_gpt.csv',
                        help='Path to output CSV file for storing extracted embeddings.')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    # Load relationship data
    relation_df = load_pandas(args.relation_data_path)
    
    # Ensure selected model is valid
    assert args.model in pricing_embeddings.keys(), 'Error! Model not present in list of models!'
    
    # Initialize OpenAIHandler with the specified model and encoding
    gpt = OpenAIHandler(model=args.model, encoding=args.encoder)
    
    # Calculate pricing if the flag is set
    if args.calculate_pricing:
        tokens_size = 0
        for i0, row in tqdm(relation_df.iterrows(), total=relation_df.shape[0], desc="Processing relationships"):
        
            tokens = [row['Title']]
            if row['Alias'] and args.include_alias: tokens += row['Alias'].split('|')
            
            # Accumulate token counts for pricing calculation
            for t0 in tokens: tokens_size += gpt.num_tokens_from_string(t0)
        
        price = pricing_embeddings[args.model]
        print()
        print(f'Estimated Price: {tokens_size/(price*1E6)}')
    
    # Extract embeddings if the flag is set
    if args.extract_embedding:
        with open(args.embedding_output_path, 'w') as f:
            f.write('Property,Embedding\n')  # CSV header
        
        for i0, row in tqdm(relation_df.iterrows(), total=relation_df.shape[0], desc="Processing relationships"):
        
            tokens = [row['Title']]
            if row['Alias'] and args.include_alias: tokens += row['Alias'].split('|')
            
            # Compute embeddings and calculate the mean embedding for the row
            embeddings = [gpt.get_embedding(t0) for t0 in tokens]
            embeddings_mean = np.array(embeddings).mean(axis=0)
            
            # Save the title and mean embedding to the CSV file
            embeddings_mean_str = ','.join(map(str, embeddings_mean))
            with open(args.embedding_output_path, 'a') as f:
                f.write(f"{row['Property']},{embeddings_mean_str}\n")