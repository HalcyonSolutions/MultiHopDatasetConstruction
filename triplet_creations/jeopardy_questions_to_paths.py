# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:24:18 2024

@author: Eduin Hernandez

Summary:
This script processes Jeopardy questions to extract meaningful paths between entities
 using the Freebase-Wikidata hybrid graph stored in Neo4j. It uses embeddings, ANN for
 nearest neighbor search, and threading to efficiently extract relationships and paths
 between nodes in the graph.

"""

import argparse
from tqdm import tqdm

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Tuple, Any

from utils.configs import global_configs
from utils.basic import load_pandas
from utils.basic import extract_literals, random_dataframes, str2bool
from utils.verify_triplets import sort_path_by_node_match, filter_tuples_by_node, visualize_path
from utils.openai_api import OpenAIHandler
from utils.fb_wiki_graph import FbWikiGraph
from utils.fb_wiki_ann import FbWikiANN

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process Jeopardy questions to extract entity paths using Neo4j graph database.")
    
    # Input Data from the CherryPicked
    parser.add_argument('--jeopardy-data-path', type=str, default='./data/jeopardy_cherrypicked.csv',
                        help='Path to the CSV file containing jeopardy questions')
    parser.add_argument('--node-data-path', type=str, default='./data/node_data_cherrypicked.csv',
                        help='Path to the CSV file containing entity data.')
    parser.add_argument('--relation-data-path', type=str, default='./data/relation_data_subgraph.csv',
                        help='Path to the CSV file containing relationship data')
    parser.add_argument('--relation-embeddings-path', type=str, default='./data/relationship_embeddings_gpt_subgraph_full.csv',
                        help='Path to the CSV file containing the relationships embeddings.')
    parser.add_argument('--database', type=str, default='subgraph',
                        help='Name of the Neo4j database to use.')
    
    # # Input Data from Jeopardy
    # parser.add_argument('--jeopardy-data-path', type=str, default='./data/jeopardy_processed.csv',
    #                     help='Path to the CSV file containing jeopardy questions')
    # parser.add_argument('--node-data-path', type=str, default='./data/node_data_fj_wiki.csv',
    #                     help='Path to the CSV file containing entity data.')
    # parser.add_argument('--relation-data-path', type=str, default='./data/relation_data_fj_wiki.csv',
    #                     help='Path to the CSV file containing relationship data')
    # parser.add_argument('--relation-embeddings-path', type=str, default='./data/relationship_embeddings_gpt_fj_wiki_full.csv',
    #                     help='Path to the CSV file containing the relationships embeddings.')
    # parser.add_argument('--database', type=str, default='fjwiki',
    #                     help='Name of the Neo4j database to use.')

    # General Parameters
    parser.add_argument('--max-relevant-relations', type=int, default=25, #25 is the ideal value
                        help='How many relevant relations to extract through nearest neighbors.')
    parser.add_argument('--max-questions', type=int, default=20,
                        help='Max number of jeopardy questions to use. For all, use None.')

    # Neo4j
    parser.add_argument('--config-path', type=str, default='./configs/configs_neo4j.ini',
                        help='Path to the configuration file for Neo4j connection.')
    # parser.add_argument('--database', type=str, default='subgraph',
    #                     help='Name of the Neo4j database to use.')
    parser.add_argument('--min-hops', type=int, default=1,
                        help='Minimum number of hops to consider in the path.')
    parser.add_argument('--max-hops', type=int, default=4,
                        help='Maximum number of hops to consider in the path.')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of workers to use for path extractions.')
    
    # ANN Parameters
    parser.add_argument('--ann-exact-computation', type=str2bool, default='True',
                        help='Flag to use exact computation for the search or an approximation.')
    parser.add_argument('--ann-nlist', type=int, default=32,
                        help='Specifies how many partitions (Voronoi cells) we’d like our ANN index to have. Used only on the approximate search.')

    # LLM models
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-small',
                        help='Model name to be used for embedding calculations (e.g., "text-embedding-3-small"). Must be a key in pricing_embeddings.')
    parser.add_argument('--encoding-model', type=str, default='cl100k_base',
                        help='Encoding name used by the model to tokenize text for embeddings.')

    # Output
    parser.add_argument('--save-results', type=str2bool, default='True', 
                        help='Flag for whether to save the results or not.')
    parser.add_argument('--result-output-path', type=str, default='./data/jeopardy_cherrypicked_path.csv',
                        help='Path to the CSV file containing jeopardy questions')

    # Statistics
    parser.add_argument('--verbose-debug', type=str2bool, default='True',
                        help='Flag to enable detailed logging for debugging purposes.')
    parser.add_argument('--verbose', type=str2bool, default='True',
                        help='Flag to enable output of summary statistics at the end of processing.')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    configs = global_configs(args.config_path)
    neo4j_parameters = configs['Neo4j']
    
    #--------------------------------------------------------------------------
    'Instantiating Models and Loading Data'
    
    # Question, Embedding, and ANN models
    embedding_gpt = OpenAIHandler(model=args.embedding_model, encoding=args.encoding_model)
    
    ann = FbWikiANN(
            data_path = args.relation_data_path,
            embedding_path = args.relation_embeddings_path, 
            exact_computation = args.ann_exact_computation,
            nlist=args.ann_nlist
            )
    
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'],
                    neo4j_parameters['password'], database = args.database)
    
    # Data
    jeopardy_df = load_pandas(args.jeopardy_data_path)
    node_data_df = load_pandas(args.node_data_path)
    relation_df = load_pandas(args.relation_data_path)
    
    node_data_df.set_index('QID', inplace=True)
    relation_df.set_index('Property', inplace=True)
    
    if 'Unnamed: 0' in jeopardy_df.columns: jeopardy_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in node_data_df.columns: node_data_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in relation_df.columns: relation_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    if args.max_questions and args.max_questions < len(jeopardy_df):
        jeopardy_df = random_dataframes(jeopardy_df, args.max_questions)
    
    noninformative_pids = ['P31', 'P279', 'P518', 'P1343']
    
    final_results = {}

    jeopardy_df['Path'] = [[] for _ in range(len(jeopardy_df))]
    jeopardy_df['has_path'] = False
    for i0, row in tqdm(jeopardy_df.iterrows(), total=len(jeopardy_df), desc="Processing Jeopardy Questions"):
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        answers = list(extract_literals(row['Answer_QID'])[0])
        q_ids = list(set(extract_literals(row['Question_QID'])[0]))
        
        log_output = []
        
        if args.verbose_debug:
            log_output.append(f'\n==================\nSample {i0+1}')
            log_output.append(f"{question}")
            log_output.append(f"Answer: {node_data_df.loc[answers[0]]['Title']}")
            log_output.append(f"Entities: {node_data_df.loc[q_ids]['Title'].tolist()}")
        
        embeddings = np.array(embedding_gpt.get_embedding(question))[None,:]
        _, indices = ann.search(embeddings, args.max_relevant_relations)
        
        p_ids        = list(set(ann.index2data(indices, 'Property', max_indices=args.max_relevant_relations)[0]))
        
        paths = []
        # question nodes and answer node
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:  # Adjust max_workers based on your system
            futures = [executor.submit(g.find_path, q0, answers[0], args.min_hops, args.max_hops, None, p_ids, noninformative_pids, True, False, False) for q0 in q_ids]

            for i1, q0 in enumerate(q_ids):
                for q1 in q_ids[i1+1:]:
                    futures.append(executor.submit(g.find_path, q0, q1, args.min_hops, args.max_hops, None, p_ids, noninformative_pids, True, False, False))
            
            # Process the completed futures as they finish
            for future in as_completed(futures):
                path_sub = future.result()
                
                path_sub = filter_tuples_by_node(path_sub, answers[0]) # throw away paths that don't have the answer node
                
                if path_sub: paths += path_sub
            
        sorted_tuples, _, _ = sort_path_by_node_match(paths, q_ids)
        
        visual_path = 'NO PATH FOUND'
        if paths:
            final_results[i0] = sorted_tuples[0]
            jeopardy_df.at[i0, 'Path'] = list(sorted_tuples[0])
            jeopardy_df.at[i0, 'has_path'] = True
            visual_path = visualize_path(sorted_tuples[0], node_data_df, relation_df)
        else:
            final_results[i0] = []
         
        if args.verbose_debug:
            log_output.append(f"{visual_path}")
            # log_output.append(f"Rels: {relation_df.loc[p_ids]['Title'].tolist()}")
            # log_output.append(f"\nDistances: {d}")
            
            # Write all log outputs at the end of the iteration
            for line in log_output:
                tqdm.write(line)


if args.save_results:
    jeopardy_df.to_csv(args.result_output_path, index=False)

if args.verbose:
    jeparday_path_df = jeopardy_df[jeopardy_df['has_path'] == True]
    for j0, row in jeparday_path_df.iterrows():
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        answers = list(extract_literals(row['Answer_QID'])[0])
        q_ids = list(set(extract_literals(row['Question_QID'])[0]))
        entities = node_data_df.loc[q_ids]['Title'].tolist()
        
        path = row['Path']
        visual_path = visualize_path(path, node_data_df, relation_df)
        
        print(f'\n==================\nSample {j0+1}')
        print(f"{question}")
        print(f"Answer: {node_data_df.loc[answers[0]]['Title']}")
        print(f"Entities: {entities}")
        print(f"{visual_path}")
    
    print('\n==================')
    print(f"Jeopardy Questions with Paths: {sum(jeopardy_df['has_path'])}/{len(jeopardy_df)}")