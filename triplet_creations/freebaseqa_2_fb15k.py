# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:01:52 2024

@author: Eduin Hernandez

Summary: Filters FreebaseQA questions to retain only those compatible with the designated Freebase
    dataset by verifying the presence of the MID entities in both the questions and the target dataset. 
    The script reads the processed Freebase questions, valid triplets, and node data,
    and outputs the filtered questions containing entities that align with the target dataset.
"""
import argparse
import pandas as pd
from utils.basic import load_triplets, load_pandas

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filters FreebaseQA questions to only use those present in the given FB15k Compatible Dataset")
    
    # Input
    parser.add_argument('--freebaseqa-path', type=str, default='./data/freebaseqa_unprocessed.csv',
                        help='Path to the CSV file containing processed Jeopardy questions with associated QIDs.')
    parser.add_argument('--triplets-path', type=str, default='./data/triplets_fb15k.txt',
                        help='Path to the text file containing valid triplets of entities used for filtering.')
    
    # Output
    parser.add_argument('--freebase-output-path', type=str, default='./data/questions/freebaseqa_fb15k.csv',
                        help='Path to save the filtered Jeopardy questions containing entities that match the target dataset.')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    fqa_df = load_pandas(args.freebaseqa_path)
    triplet_df = load_triplets(args.triplets_path)
    
    valid_nodes = set(triplet_df['head']) | set(triplet_df['tail'])
    valid_rels = set(triplet_df['relation'])
    
    valid_df = fqa_df[fqa_df['TopicEntityMid'].isin(valid_nodes) & fqa_df['AnswersMid'].isin(valid_nodes)]
    valid_df.loc[:, 'InferentialChain'] = valid_df['InferentialChain'].str.replace('..', ';/').str.replace('.', '/').str.replace(';', '.')
    valid_df.loc[:, 'InferentialChain'] = valid_df['InferentialChain'].apply(lambda x: '/' + x if not x.startswith('/') else x)
    valid_df = valid_df[valid_df['InferentialChain'].isin(valid_rels)]
    
    valid_question_id = set(valid_df['Question-Number'])
    
    print(f'Number of Questions: {len(valid_question_id)}')
    
    # Add 'relevant-entities' column aggregating 'TopicEntityMid' per 'Question-Number'
    relevant_entities_df = valid_df.groupby('Question-Number')['TopicEntityMid'].apply(lambda x: list(set(x))).reset_index()
    relevant_rel_df = valid_df.groupby('Question-Number')['InferentialChain'].apply(lambda x: list(set(x))).reset_index()
    valid_df = valid_df.merge(relevant_entities_df, on='Question-Number', suffixes=('', '_relevant'))
    valid_df.rename(columns={'TopicEntityMid_relevant': 'Relevant-Entities'}, inplace=True)
    valid_df = valid_df.merge(relevant_rel_df, on='Question-Number', suffixes=('', '_relevant'))
    valid_df.rename(columns={'InferentialChain_relevant': 'Relevant-Relations'}, inplace=True)
    valid_df.rename(columns={'AnswersMid': 'Answer-Entity'}, inplace=True)
    
    valid_df.to_csv(args.freebase_output_path, index=False)
    
    # # Retains only the Question-Number, Question, and Answer
    qa_only_df = valid_df.drop_duplicates(subset='Question-Number')[['Question-Number', 'Question', 'Answer', 'Relevant-Entities', 'Relevant-Relations', 'Answer-Entity']]
    qa_only_df.to_csv(args.freebase_output_path.replace('.csv','') + '_clean.csv', index=False)
