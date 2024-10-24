# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:04:38 2024

@author: Eduin Hernandez

Summary: This script processes Jeopardy questions to extract relevant RDF data,
 identify entities using spaCy, and create a subgraph of neighboring nodes using
 Neo4j and FbWikiGraph. Key components include multi-threaded data processing,
 entity extraction, neighborhood extraction, and saving processed data for analysis.
 
TODO: 1) Make functions in separate file as a pacakge
TODO: 2) Create an argument parser for this file
"""

import pandas as pd
import spacy
import itertools

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.basic import load_pandas, save_set_pandas, sort_by_qid
from utils.wikidata_v2 import retry_fetch, search_wikidata_relevant_id

# Function to process each row and update RDF columns
def process_row(nlp, row):
    idx, data = row
    
    # Extract values
    category = data['Category'].replace('(', '').replace(')', '').strip()
    question = data['Question'].replace('(', '').replace(')', '').strip()
    answer = data['Answer'].replace('(', '').replace(')', '').strip()
    
    # Search for RDF IDs
    answer_tokens = list(set(split_entities(answer) + [answer]))
    answer_ids = [rdf_id for ent in answer_tokens if (rdf_id := search_wikidata_relevant_id(ent))]
    # answer_rdf = search_wikidata_relevant_id(answer)
    
    # Process the question with spaCy to identify entities
    doc = nlp(category + '. ' + question)
    entities = extract_entities(doc)

    # Search for RDF IDs of entities, avoiding calling the function twice
    question_ids = [rdf_id for ent in entities if (rdf_id := search_wikidata_relevant_id(ent))]

    # Return updated data
    return idx, question_ids, answer_ids

# Function to update the DataFrame with the results
def update_dataframe(df, results):
    for result in results:
        idx, question_ids, answer_rdf = result
        df.at[idx, 'Question_RDF'] = question_ids
        df.at[idx, 'Answer_RDF'] = answer_rdf

# Wrapper function to handle retry fetch in threads
def process_row_with_retry(nlp, row):
    return retry_fetch(process_row, nlp, row, max_retries=3, timeout=10, verbose=True)

def split_entities(text: str) -> list:
    entities = []
    # If both ', ' and ' & ' are present in the entity
    if '&' in text and ', ' in text:
        # Split based on ', ' first
        sub_entities = text.split(', ')
        for sub_ent in sub_entities:
            if ' & ' in sub_ent:
                # Add the original part containing '&'
                entities.append(sub_ent)
                # Split further on ' & ' and add individual parts
                entities.extend(sub_ent.split(' & '))
            else:
                entities.append(sub_ent)
    # If only ' & ' is present
    elif '&' in text:
        # Keep the original entity
        entities.append(text)
        # Also split based on ' & ' and add the individual entities
        entities.extend(text.split(' & '))
    # If only ', ' is present
    elif ', ' in text:
        # Split based on ', ' and add them separately
        entities.extend(text.split(', '))
    # No special characters, keep the entity as is
    else:
        entities.append(text)
    return entities

# Function to extract entities
def extract_entities(doc):
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['CARDINAL', 'QUANTITY', 'ORDINAL']:
            continue
        
        entities += split_entities(ent.text)
        
    return entities


if __name__ == '__main__':
    'Input'
    jeopardy_data_path = './data/jeopardy_bojan.csv'
    fbwiki_data_path = './data/rdf_data.csv'
    
    # Load the Data
    jeopardy_data = load_pandas(jeopardy_data_path)

    # Strip any extra spaces in the column names for safety
    jeopardy_data.columns = jeopardy_data.columns.str.strip()

    # remove images and links
    jeopardy_data = jeopardy_data[~jeopardy_data['Question'].str.contains('href', case=False, na=False)]

    #--------------------------------------------------------------------------
    'Spacy'
    nlp = spacy.load("en_core_web_sm")

    #--------------------------------------------------------------------------
    
    jeopardy_data['Question_RDF'] = [None] * len(jeopardy_data)
    jeopardy_data['Answer_RDF'] = [None] * len(jeopardy_data)
    
    # 'Single Threaded'
    # for row in jeopardy_data.iterrows():
    #     process_row(nlp, row)
    
    'Multi-Threaded'
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:  # Adjust the number of workers based on your CPU cores
        futures = [executor.submit(process_row_with_retry, nlp, row) for row in jeopardy_data.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Jeopardy Data'):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing row: {e}")
    
    # Update the DataFrame with the results
    update_dataframe(jeopardy_data, results)
    
    jeopardy_data.to_csv('./data/jeopardy_unprocessed.csv', index=False)
    
    #--------------------------------------------------------------------------
    filtered_jeopardy_data = jeopardy_data[
        (jeopardy_data['Answer_RDF'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)) &  # Remove rows where 'Answer_RDF' is None
        (jeopardy_data['Question_RDF'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False))  # Remove rows where 'Question_RDF' is an empty list and has at least 2 QIDs
    ]
    
    # Optional: Save the result to a new CSV file
    filtered_jeopardy_data.to_csv('./data/jeopardy_processed.csv', index=False)
    #--------------------------------------------------------------------------
    jeo_set = set(itertools.chain(*filtered_jeopardy_data['Answer_RDF'].tolist()))
    jeo_set.update(set(itertools.chain(*filtered_jeopardy_data['Question_RDF'].tolist())))
    if '' in jeo_set: jeo_set.remove('')
    
    save_set_pandas(jeo_set, './data/nodes_jeopardy.txt')
