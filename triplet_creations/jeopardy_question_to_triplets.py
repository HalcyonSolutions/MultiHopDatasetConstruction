# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:56:54 2024

@author: Eduin Hernandez
"""
import argparse

import numpy as np
import pandas as pd

from utils.basic import load_pandas,  load_triplets
from utils.basic import extract_literals, random_dataframes, str2bool
from utils.openai_api import OpenAIHandler
from utils.fb_wiki_ann import FbWikiANN
from utils.verify_triplets import map_triplet_titles, confirm_triplets, is_answerable
from utils.question_triplets import prepare_prompt, extract_triplets, titles2ids

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="")
    
    # Input Data from the CherryPicked
    parser.add_argument('--jeopardy-data-path', type=str, default='./data/jeopardy_cherrypicked.csv',
                        help='Path to the CSV file containing jeopardy questions')
    parser.add_argument('--node-data-path', type=str, default='./data/node_data_cherrypicked.csv',
                        help='Path to the CSV file containing entity data.')
    parser.add_argument('--triplets-path', type=str, default='./data/triplets_fj_wiki.txt',
                        help='Path to the CSV file containing the entire triplet set.')
    parser.add_argument('--relation-data-path', type=str, default='./data/relation_data_subgraph.csv',
                        help='Path to the CSV file containing relationship data')
    parser.add_argument('--relation-embeddings-path', type=str, default='./data/relationship_embeddings_gpt_subgraph_full.csv',
                        help='Path to the CSV file containing the relationships embeddings.')
    
    # # Input Data from Jeopardy
    # parser.add_argument('--jeopardy-data-path', type=str, default='./data/jeopardy_processed.csv',
    #                     help='Path to the CSV file containing jeopardy questions')
    # parser.add_argument('--node-data-path', type=str, default='./data/node_data_fj_wiki.csv',
    #                     help='Path to the CSV file containing entity data.')
    # parser.add_argument('--triplets-path', type=str, default='./data/triplets_fj_wiki.txt',
    #                     help='Path to the CSV file containing the entire triplet set.')
    # parser.add_argument('--relation-data-path', type=str, default='./data/relation_data_fj_wiki.csv',
    #                     help='Path to the CSV file containing relationship data')
    # parser.add_argument('--relation-embeddings-path', type=str, default='./data/relationship_embeddings_gpt_fj_wiki_full.csv',
    #                     help='Path to the CSV file containing the relationships embeddings.')

    # General Parameters
    parser.add_argument('--max-relevant-relations', type=int, default=15,
                        help='How many relevant relations to extract through nearest neighbors.')
    parser.add_argument('--max-questions', type=int, default=20,
                        help='Max number of jeopardy questions to use. For all, use None.')

    # ANN Parameters
    parser.add_argument('--ann-exact-computation', type=str2bool, default='True',
                        help='Flag to use exact computation for the search or an approximation.')
    parser.add_argument('--ann-nlist', type=int, default=32,
                        help='Specifies how many partitions (Voronoi cells) we’d like our ANN index to have. Used only on the approximate search.')

    # LLM models
    parser.add_argument('--question-model', type=str, default='gpt-4o-mini',
                        help='Model name to perform question queries (i.e., "gpt-4o-mini"). Must be a key in pricing_embeddings. Must be a key in pricing_keys.')
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-small',
                        help='Model name to be used for embedding calculations (e.g., "text-embedding-3-small"). Must be a key in pricing_embeddings.')
    parser.add_argument('--encoding-model', type=str, default='cl100k_base',
                        help='Encoding name used by the model to tokenize text for embeddings.')
    
    # Output Path
    parser.add_argument('--save-results', type=str2bool, default='True', 
                        help='Flag for whether to save the results or not.')
    parser.add_argument('--result-output-path', type=str, default='./data/jeopardy_cherrypicked_path.csv',
                        help='Path to the CSV file containing jeopardy questions')

    # Statistics
    parser.add_argument('--verbose', type=str2bool, default='True')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    'Instantiating Models and Loading Data'
    
    # Question, Embedding, and ANN models
    chat_gpt = OpenAIHandler(model=args.question_model)
    embedding_gpt = OpenAIHandler(model=args.embedding_model, encoding=args.encoding_model)
    
    ann = FbWikiANN(
            data_path = args.relation_data_path,
            embedding_path = args.relation_embeddings_path, 
            exact_computation = args.ann_exact_computation,
            nlist=args.ann_nlist
            )
    # Data
    jeopardy_df = load_pandas(args.jeopardy_data_path)
    node_data_df = load_pandas(args.node_data_path)
    relation_df = load_pandas(args.relation_data_path)
    triplet_df = load_triplets(args.triplets_path)

    if 'Unnamed: 0' in node_data_df.columns: node_data_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in jeopardy_df.columns: jeopardy_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in relation_df.columns: relation_df.drop(columns=['Unnamed: 0'], inplace=True)
    if args.max_questions and args.max_questions < len(jeopardy_df):
        jeopardy_df = random_dataframes(jeopardy_df, args.max_questions)


    #--------------------------------------------------------------------------
    
    new_row = pd.DataFrame([{'QID': 'Unknown', 'Title': 'Unknown'},
                            {'QID': 'Unknown', 'Title': 'Unknown Key'},
                            {'QID': 'Unknown', 'Title': '(Unknown)'},
                            {'QID': 'Unknown', 'Title': '(Unknown Key)'}])
    
    triplet_list = triplet_df.values.tolist()
    triplet_set = set(tuple(t[:3]) for t in triplet_list)  # Create a set for quick lookup
    
    jeopardy_df['Triplets'] = [[] for _ in range(len(jeopardy_df))]
    jeopardy_df['QIDs'] = [[] for _ in range(len(jeopardy_df))]
    jeopardy_df['Pids'] = [[] for _ in range(len(jeopardy_df))]
    jeopardy_df['has_triplets'] = False
    jeopardy_df['is_answerable'] = False
    
    for i0, row in jeopardy_df.iterrows():
        print(f'==================\nSample {i0+1}')
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        answers = list(extract_literals(row['Answer_QID'])[0])
        
        q_ids = set(extract_literals(row['Question_QID'])[0])
        
        embeddings = np.array(embedding_gpt.get_embedding(question))[None,:]
        _, indices = ann.search(embeddings, args.max_relevant_relations)
        
        prop_title  = ann.index2data(indices, 'Title', max_indices=args.max_relevant_relations)[0]
        p_ids        = set(ann.index2data(indices, 'Property', max_indices=args.max_relevant_relations)[0])
        descriptions = ann.index2data(indices, 'Description', max_indices=args.max_relevant_relations)[0]
        
        named_entities = ['Unknown Key'] + node_data_df[node_data_df['QID'].isin(q_ids)]['Title'].tolist()
        prompt = prepare_prompt(question=question, entities=named_entities, relations=prop_title, descriptions=descriptions)
        response = chat_gpt.query(prompt)
        
        print(f"Answer: {response['answer']}")
        
        triplets = extract_triplets(response['answer'])
        
        df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])
        
        names = set(df['head'].tolist()) | set(df['tail'].tolist())
        names_df = node_data_df[node_data_df['Title'].isin(names)]
        if len(names_df)>0: q_ids =  q_ids | set(names_df['QID'].tolist())
        
        df = titles2ids(df, node_data_df, relation_df, new_row, q_ids, p_ids)
        
        confirmed = confirm_triplets(df, triplet_set, triplet_list)
        confirmed_df = pd.DataFrame(confirmed, columns=['head', 'relation', 'tail'])
        
        answerable = False
        for ans in answers:
            answerable |= is_answerable(confirmed_df, triplet_df, ans)
        
        rels = list(set(confirmed_df['relation'].tolist()))
        nodes = list((set(confirmed_df['head'].tolist()) | set(confirmed_df['tail'].tolist())) - set(['Unknown']))
        jeopardy_df.at[i0, 'Triplets'] = confirmed
        jeopardy_df.at[i0, 'Pids'] = rels
        jeopardy_df.at[i0, 'QIDs'] = nodes
        jeopardy_df.at[i0, 'has_triplets'] = bool(confirmed)
        jeopardy_df.at[i0, 'is_answerable'] = answerable
        
        if args.verbose:
            print('GPT Triplets')
            if not(triplets):
                print('\tEMPTY')
            for r in triplets:
                print(f"\t[{r[0]}, {r[1]}, {r[2]}]")
                
            print('Valid Triplets')
            if not(confirmed):
                print('\tEMPTY')
            for _, r in confirmed_df.iterrows():
                print(f"\t[{r['head']}, {r['relation']}, {r['tail']}]")
            print(f'Answerable: {answerable}' )

if args.save_results:
    jeopardy_df.to_csv(args.result_output_path, index=False)

if args.verbose:
    print('\n==================')
    print(f"Answerable Questions: {sum(jeopardy_df['is_answerable'])}/{len(jeopardy_df)}")
