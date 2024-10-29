# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:46:41 2024

@author: Eduin Hernandez
"""

from utils.basic import load_pandas, extract_literals, load_triplets
from utils.openai_api import OpenAIHandler
from utils.fb_wiki_ann import FbWikiANN
import re
import numpy as np
import pandas as pd

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

def extract_triplets(response):
    triplets = re.findall(r'\[(.*?)\]', response)
    result = [triplet.split(', ') for triplet in triplets]
    return result

def map_triplet_titles(tripled_df: pd.DataFrame, relation_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    rdf_to_title_map = node_df.copy()
    rdf_to_title_map = rdf_to_title_map.set_index('RDF')['Title']
    
    prop_to_title_map = relation_df.set_index('Property')['Title']
    tripled_df[['head', 'tail']] = tripled_df[['head', 'tail']].apply(lambda col: col.map(rdf_to_title_map).fillna(col))
    tripled_df[['relation']] = tripled_df[['relation']].apply(lambda col: col.map(prop_to_title_map).fillna(col))
    return tripled_df

def triplet_confirmation(df, triplet_set, triplet_list):
    confirmed_triplets = []
    for _, row in df.iterrows():
        triplet = (row['head'], row['relation'], row['tail'])
        reverse_triplet = (row['tail'], row['relation'], row['head'])
        if 'Unknown' in [row['head'], row['tail']]:
            unknown_tail = row['tail'] == 'Unknown'
            partial_triplet = [row['head'], row['relation']] if unknown_tail else [row['tail'], row['relation']]
            matching_triplet = [t for t in triplet_list if t[:2] == partial_triplet] if unknown_tail else [t for t in triplet_list if t[1:] == partial_triplet]
            matching_triplet_reverse = [t for t in triplet_list if t[1:] == partial_triplet] if unknown_tail else [t for t in triplet_list if t[:2] == partial_triplet]
            if matching_triplet:
                confirmed_triplets.append([row['head'], row['relation'], row['tail']])
            if matching_triplet_reverse:
                confirmed_triplets.append([row['tail'], row['relation'], row['head']])
        elif triplet in triplet_set:
            confirmed_triplets.append([row['head'], row['relation'], row['tail']])
        elif reverse_triplet in triplet_set:
            confirmed_triplets.append([row['tail'], row['relation'], row['head']])
    
    # Remove duplicate triplets
    confirmed_triplets = [list(t) for t in set(tuple(t) for t in confirmed_triplets)]
    return confirmed_triplets

def is_answerable(df, triplet_df, answer):
    unknown_df = df[
        df['head'].eq('Unknown') | df['tail'].eq('Unknown')
    ].copy()
    unknown_df[['head', 'tail']] = unknown_df[['head', 'tail']].replace('Unknown', answer)
    
    # Check if any triplet in unknown_df exists in triplet_df
    unknown_df['exists'] = unknown_df.apply(lambda x: tuple(x) in triplet_df[['head', 'relation', 'tail']].itertuples(index=False, name=None), axis=1)
    return unknown_df['exists'].any()
    
if __name__ == '__main__':
    #* Start openai handler
    chat_gpt = OpenAIHandler(model='gpt-4o-mini')
    embedding_gpt = OpenAIHandler(model='text-embedding-3-small', encoding='cl100k_base')
    
    jeopardy_df = load_pandas('./data/jeopardy_cherrypicked.csv')
    node_data_df = load_pandas('./data/node_data_cherrypicked.csv').drop(columns=['Unnamed: 0'])
    relation_df = load_pandas('./data/relation_data_subgraph.csv')
    triplet_df = load_triplets('./data/triplets_fj_wiki.txt')

    ann = FbWikiANN(
            data_path = './data/relation_data_subgraph.csv',
            embedding_path = './data/relationship_embeddings_gpt_subgraph.csv', 
            exact_computation = True,
            nlist=32
            )
    
    guesses = 10
    valid_triplets = []
    extracted_relations = []
    extracted_triplets = []
    extracted_entities = []
    new_row = pd.DataFrame([{'RDF': 'Unknown', 'Title': 'Unknown'},
                            {'RDF': 'Unknown', 'Title': 'Unknown Key'},
                            {'RDF': 'Unknown', 'Title': '(Unknown)'},
                            {'RDF': 'Unknown', 'Title': '(Unknown Key)'}])
    triplet_list = triplet_df.values.tolist()
    triplet_set = set(tuple(t[:3]) for t in triplet_list)  # Create a set for quick lookup
    for i0, row in jeopardy_df.iterrows():
        print(f'==================\nSample {i0+1}')
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        q_rdfs = set(extract_literals(row['Question_RDF'])[0])
        
        named_entities = ['Unknown Key'] + node_data_df[node_data_df['RDF'].isin(q_rdfs)]['Title'].tolist()
        prompt = get_prompt(entities=named_entities, question=question)
        response = chat_gpt.query(prompt)
        
        matches = list(set(find_relations(response['answer'])))
        embeddings = np.array([embedding_gpt.get_embedding(t0) for t0 in matches])
        _, indices = ann.search(embeddings, guesses)
        
        # prop_title  = ann.index2data(indices, 'Title', max_indices=guesses)
        prop        = ann.index2data(indices, 'Property', max_indices=guesses)
        extracted_relations.append(matches)
        
        match_dict = {'<' + m + '>': prop[i] for i, m in enumerate(matches)}
            
        #----------------------------------------------------------------------
        # Extract and process triplets
        triplets = extract_triplets(response['answer'])
        df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])
        extracted_triplets.append(triplets)
        
        names = set(df['head'].tolist()) | set(df['tail'].tolist())
        names_df = node_data_df[node_data_df['Title'].isin(names)]
        if len(names_df)>0: q_rdfs =  q_rdfs | set(names_df['RDF'].tolist())
        extracted_entities.append(q_rdfs)
        
        valid_nodes = node_data_df[node_data_df['RDF'].isin(q_rdfs)].copy()
        valid_nodes = pd.concat([valid_nodes, new_row], ignore_index=True).fillna('')
        valid_nodes_map = valid_nodes.set_index('Title')['RDF']
        
        # Replace head and tail with RDF values
        df['head'] = df['head'].map(valid_nodes_map).fillna(df['head'])
        df['tail'] = df['tail'].map(valid_nodes_map).fillna(df['tail'])
        
        # Drop any head or tail that do not start with 'Q' or 'Unknown'
        df = df[df['head'].str.startswith(('Q', 'Unknown')) & df['tail'].str.startswith(('Q', 'Unknown'))]

        # Drop rows where both head and tail are 'Unknown'
        df = df[~((df['head'] == 'Unknown') & (df['tail'] == 'Unknown'))]
        
        # Replace relations in the triplet DataFrame using match_dict
        df['relation'] = df['relation'].apply(lambda x: match_dict.get(x, [x]))
        df = df.explode('relation')

        # Remove duplicate triplets
        df.drop_duplicates(inplace=True)
        
        # # Confirm existence conditions for triplets
        # confirmed_triplets = []
        # for _, row in df.iterrows():
        #     triplet = (row['head'], row['relation'], row['tail'])
        #     reverse_triplet = (row['tail'], row['relation'], row['head'])
        #     if 'Unknown' in [row['head'], row['tail']]:
        #         unknown_tail = row['tail'] == 'Unknown'
        #         partial_triplet = [row['head'], row['relation']] if unknown_tail else [row['tail'], row['relation']]
        #         matching_triplet = [t for t in triplet_list if t[:2] == partial_triplet] if unknown_tail else [t for t in triplet_list if t[1:] == partial_triplet]
        #         matching_triplet_reverse = [t for t in triplet_list if t[1:] == partial_triplet] if unknown_tail else [t for t in triplet_list if t[:2] == partial_triplet]
        #         if matching_triplet:
        #             confirmed_triplets.append([row['head'], row['relation'], row['tail']])
        #         if matching_triplet_reverse:
        #             confirmed_triplets.append([row['tail'], row['relation'], row['head']])
        #     elif triplet in triplet_set:
        #         confirmed_triplets.append([row['head'], row['relation'], row['tail']])
        #     elif reverse_triplet in triplet_set:
        #         confirmed_triplets.append([row['tail'], row['relation'], row['head']])
        
        # # Remove duplicate triplets
        # confirmed_triplets = [list(t) for t in set(tuple(t) for t in confirmed_triplets)]
        
        confirmed_triplets = triplet_confirmation(df, triplet_set, triplet_list)
        valid_triplets.append(confirmed_triplets)

answerable_qty = 0
for i0, v0 in enumerate(valid_triplets):
    row = jeopardy_df.iloc[i0]
    answer = list(extract_literals(row['Answer_RDF'])[0])[0]
    question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
    named_entities = ['Unknown Key'] + node_data_df[node_data_df['RDF'].isin(extracted_entities[i0])]['Title'].tolist()
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