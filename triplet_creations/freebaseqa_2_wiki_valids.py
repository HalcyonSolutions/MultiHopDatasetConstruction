# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:05:16 2024

@author: Eduin Hernandez

Summary: Filters FreebaseQA questions to retain only those compatible with the designated wikidata
    dataset by verifying the presence of entities in both the questions and the target dataset. 
    The script reads the processed Freebase questions, valid triplets, and node data,
    and outputs the filtered questions containing entities that align with the target dataset.
"""
import argparse

import pandas as pd

from utils.basic import load_pandas, load_triplets


def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="")
    
    # Input
    parser.add_argument('--freebaseqa-path', type=str, default='./data/freebaseqa_unprocessed.csv',
                        help='Path to the CSV file containing processed Freebase questions with associated MIDs.')
    parser.add_argument('--triplets-path', type=str, default='./data/link_prediction/Fb-Wiki/triplets.txt',
                        help='Path to the text file containing valid triplets of entities used for filtering.')
    parser.add_argument('--nodes-data-path', type=str, default='./data/metadata/node_data_fb_wiki.csv',
                        help='Path to the CSV file containing node data, including the respective entity name for each QID.')
    
    # Output
    parser.add_argument('--freebase-output-path', type=str, default='./data/qa/FreebaseQA/freebase_fb_wiki.csv',
                        help='Path to save the filtered Jeopardy questions containing entities that match the target dataset.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    freebase_df = load_pandas(args.freebaseqa_path)
    triplets = load_triplets(args.triplets_path)
    node_data = load_pandas(args.nodes_data_path)
    
    nodes = set(triplets['head'].tolist()) | set(triplets['tail'].tolist())
    node_data = node_data[node_data['QID'].isin(nodes) &
                          (node_data['MID'] != '')]

    node_data = node_data.drop_duplicates(subset='MID', keep='first')
    node_data = node_data.set_index('MID')
    
    nodes = set(node_data.index.tolist())

    valid_df = freebase_df[freebase_df['TopicEntityMid'].isin(nodes) & freebase_df['AnswersMid'].isin(nodes)]
    valid_question_id = set(valid_df['Question-Number'])
    
    # Create a new DataFrame to store filtered rows
    filtered_rows = []
    
    for j, (i0, row) in enumerate(valid_df.iterrows()):
        q_node = node_data.loc[row['TopicEntityMid']]
        a_node = node_data.loc[row['AnswersMid']]
        
        # Update q_qid with the intersection list
        row_copy = row.copy()
        
        row_copy.loc['Question-QID'] = q_node['QID']
        row_copy.loc['Answer-QID'] = a_node['QID']
        row_copy.loc['Question-Entities'] = q_node['Title']
        row_copy.loc['Answer-Entities'] = a_node['Title']
        
        filtered_rows.append(row_copy)
    
    # Create a new DataFrame with filtered rows
    filtered_df = pd.DataFrame(filtered_rows)
    
    filtered_df.to_csv(args.freebase_output_path, index=False)
    
    # Retains only the Question-Number, Question, and Answer
    qa_only_df = filtered_df.drop_duplicates(subset='Question-Number')[['Question-Number', 'Question', 'Answer']]
    qa_only_df.to_csv(args.freebase_output_path.replace('.csv','') + '_clean.csv', index=False)