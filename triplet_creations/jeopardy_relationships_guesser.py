# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:46:41 2024

@author: Eduin Hernandez
"""
import argparse

import re
import numpy as np
import pandas as pd

from utils.basic import load_pandas,  load_triplets
from utils.basic import extract_literals, random_dataframes, str2bool
from utils.openai_api import OpenAIHandler
from utils.fb_wiki_ann import FbWikiANN
from utils.verify_triplets import map_triplet_titles, confirm_triplets, is_answerable
from utils.question_triplets import extract_triplets, titles2ids

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="")
    
    # Input Data
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

    return parser.parse_args()

def get_prompt(entities, question):
    prompt = f"""
    ***Jeopardy Evaluation***
    Below, I will provide you with:
    1. A set of questions.
    2. A list of names
    Your task is to extract all relationships from each question, following these guidelines:
    1. **Triplet Structure**: Format each triplet as [entity1, relationship, entity2], where:
    - `entity1` and `entity2` are entities derived from the question or the provided list.
    - The `relationship` must be a term or concept explicitly or implicitly present in the question.
    - The `relationship` can have multiple names or variations, so provide as many as possible.
    2. **Handling Irrelevant or Missing Entities**:
    - If an entity from the provided list is not relevant to the context of the question, you may exclude it.
    - The exception is the (Unknown Key), which must always be included in at least one triplet as a meaningful placeholder for an unknown node.
    3. **(Unknown Key) as Placeholder**:
    - Treat (Unknown Key) as a placeholder for an unknown entity.
    - Do not replace (Unknown Key) with any known entity, but ensure it connects logically to other entities or relationships in the question.
    - (Unknown Key) cannot be both entity1 and entity2 in the same triplet.
    4. **Extracting Logical Relationships**:
    - Extract relationships that form logical and contextually relevant connections between entities.
    - Ensure relationships are meaningful in the context of the given question.
    
    **Response Format**:
    Return the relationships in a comma-separated format.
    Only return this line below:
    `Only return the list of relationships, without any additional text or formatting. Relationships have to be placed within angle brackets <>.
    i.e.: [entity1, <relationship>, entity2], [entity1, <relationship>, entity2] .... and so on [entity1N , <relationship>, entity2N]`
    ---
    Question: {question}
    Entities: {entities}
    """

    return prompt

def find_relations(response):
    matches = re.findall(r'<(.*?)>', response)
    return matches
    
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
    if args.max_questions and args.max_questions < len(jeopardy_df):
        jeopardy_df = random_dataframes(jeopardy_df, args.max_questions)
    
    #--------------------------------------------------------------------------
    
    valid_triplets = []
    extracted_relations = []
    extracted_triplets = []
    extracted_entities = []
    new_row = pd.DataFrame([{'QID': 'Unknown', 'Title': 'Unknown'},
                            {'QID': 'Unknown', 'Title': 'Unknown Key'},
                            {'QID': 'Unknown', 'Title': '(Unknown)'},
                            {'QID': 'Unknown', 'Title': '(Unknown Key)'}])
    
    triplet_list = triplet_df.values.tolist()
    triplet_set = set(tuple(t[:3]) for t in triplet_list)  # Create a set for quick lookup
    for i0, row in jeopardy_df.iterrows():
        print(f'==================\nSample {i0+1}')
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        q_ids = set(extract_literals(row['Question_QID'])[0])
        
        named_entities = ['Unknown Key'] + node_data_df[node_data_df['QID'].isin(q_ids)]['Title'].tolist()
        prompt = get_prompt(entities=named_entities, question=question)
        response = chat_gpt.query(prompt)
        
        matches = list(set(find_relations(response['answer'])))
        embeddings = np.array([embedding_gpt.get_embedding(t0) for t0 in matches])
        _, indices = ann.search(embeddings, args.max_relevant_relations)
        
        # prop_title  = ann.index2data(indices, 'Title', max_indices=guesses)
        p_ids        = ann.index2data(indices, 'Property', max_indices=args.max_relevant_relations)
        extracted_relations.append(matches)
        
        match_dict = {'<' + m + '>': p_ids[i] for i, m in enumerate(matches)}
            
        #----------------------------------------------------------------------
        # Extract and process triplets
        triplets = extract_triplets(response['answer'])
        df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])
        extracted_triplets.append(triplets)
        
        names = set(df['head'].tolist()) | set(df['tail'].tolist())
        names_df = node_data_df[node_data_df['Title'].isin(names)]
        if len(names_df)>0: q_ids =  q_ids | set(names_df['QID'].tolist())
        extracted_entities.append(q_ids)
        
        # Replace relations in the triplet DataFrame using match_dict
        df['relation'] = df['relation'].apply(lambda x: match_dict.get(x, [x]))
        df = df.explode('relation')
        
        df = titles2ids(df, node_data_df, relation_df, new_row, q_ids, p_ids)
        
        confirmed = confirm_triplets(df, triplet_set, triplet_list)
        valid_triplets.append(confirmed)

answerable_qty = 0
for i0, v0 in enumerate(valid_triplets):
    row = jeopardy_df.iloc[i0]
    answer = list(extract_literals(row['Answer_QID'])[0])[0]
    question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
    named_entities = ['Unknown Key'] + node_data_df[node_data_df['QID'].isin(extracted_entities[i0])]['Title'].tolist()
    guessed_relations = extracted_relations[i0]
    guessed_triplets = extracted_triplets[i0]
    # Update DataFrame with confirmed triplets
    confirmed_triplets_df = pd.DataFrame(v0, columns=['head', 'relation', 'tail'])
    
    answerable = is_answerable(confirmed_triplets_df, triplet_df, answer)
    answerable_qty += answerable
    
    confirmed_triplets_df = map_triplet_titles(confirmed_triplets_df, relation_df, node_data_df)
    
    print(f'==================\nSample {i0+1}')
    print(f'Sentence: {question}')
    print(f'Entities: {named_entities}')
    print(f'Extracted Relations: {guessed_relations}')
    print('Extracted Triplets')
    if not(guessed_triplets):
        print('\tEMPTY')
    for row in guessed_triplets:
        print(f"\t[{row[0]}, {row[1]}, {row[2]}]")
    print('Valid Triplets')
    if not(v0):
        print('\tEMPTY')
    for _, row in confirmed_triplets_df.iterrows():
        print(f"\t[{row['head']}, {row['relation']}, {row['tail']}]")
    print(f'Answerable: {answerable}' )
        
print('\n==================')
print(f'Answerable Questions: {answerable_qty}/{len(valid_triplets)}')