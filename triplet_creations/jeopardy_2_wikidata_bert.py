# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:18:54 2024

@author: Eduin Hernandez

Summary: Uses Named Entity Extraction via a BERT model on the questions provided
          and maps to the closest entity in Wikidata using relevancy search and
          embedding distance for disambiguation. The script reads Jeopardy questions,
          processes them with NLP models, and outputs enriched questions with linked entities.
"""
import argparse
from typing import Tuple, List, Set

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
from nltk.corpus import stopwords

import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from utils.basic import load_pandas, str2bool, random_dataframes, extract_literals
from utils.openai_api import OpenAIHandler
from utils.nlp_ner import capitalize, split_entities, guess_wiki_entity, remove_duplicate_entities, extract_entities
from utils.wikidata_v2 import retry_fetch


def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Uses BERT-based Named Entity Recognition and embedding-based search to link entities in Jeopardy questions to Wikidata.")
    
    'Input'
    parser.add_argument('--jeopardy-path', type=str, default='./data/jeopardy_bojan.csv',
                        help='Path to the CSV file containing the original Jeopardy questions')
    
    parser.add_argument('--stage-one', type=str2bool, default='True',
                        help='Boolean flag to indicate whether to execute the first stage of processing which includes entity extraction.')
    parser.add_argument('--stage-two', type=str2bool, default='False',
                        help='Boolean flag to indicate whether to execute the second stage of processing which includes filtering and saving results.')
    
    parser.add_argument('--use-openai', type=str2bool, default='False',
                        help='Boolean flag to indicate whethe to use openai embedding models (True) or Hugging Face (False).')
    
    parser.add_argument('--max-questions', type=int, default=100,
                        help='Maximum number of Jeopardy questions to process. Use None to process all.')
    parser.add_argument('--max-workers', type=int, default=15,
                    help='Maximum number of worker threads to use for multi-threaded processing.')
    parser.add_argument('--max-relevance', type=int, default=3,
                        help='Maximum number of relevance hits per token during entity linking.')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducibility when selecting a subset of questions. Use None to not use randomization.')

    # Openai
    parser.add_argument('--oa-embedding-model', type=str, default='text-embedding-3-small',
                        help='OpenAi model name to be used for embedding calculations (e.g., "text-embedding-3-small"). Must be a key in pricing_embeddings.')
    parser.add_argument('--oa-encoding-model', type=str, default='cl100k_base',
                        help='Encoding name used by the model to tokenize text for embeddings.')
    
    # Hugging Face
    parser.add_argument('--hh-embedding-model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='Hugging Face model to use for sentence embeddings.')
    parser.add_argument('--hh-ner-model', type=str, default='dslim/bert-base-NER',
                        help='Hugging Face model to use for Named Entity Recognition (NER).')
    parser.add_argument('--aggregation-strategy', type=str, default='simple',
                        help='Aggregation strategy to use for NER (e.g., "simple", "first", "average").')

 
    'Output'
    parser.add_argument('--jeopardy-unprocessed-path', type=str, default='./data/jeopardy_unprocessed_bert.csv',
                        help='Path to save the Jeopardy questions after the initial entity linking stage.')
    parser.add_argument('--jeopardy-processed-path', type=str, default='./data/jeopardy_processed_bert.csv',
                        help='Path to save the final filtered Jeopardy questions after processing.')
    
    return parser.parse_args()

def process_row(row: Tuple[int, dict], nlp, embedder, args,
                stop_words: Set[str] = set()) -> Tuple[int, List[dict], List[dict]]:
    
    idx, data = row
    
    # Extract and clean the category, question, and answer text
    category = capitalize(data['Category'])
    question = data['Question']
    answer = data['Answer']
    
    # Formulate the sentence to be used for embedding
    sentence = f'Category: {category} Question: {question} Answer: {answer}'
    
    # Compute the embedding for the entire sentence
    sentence_embedding = embedder.encode(sentence) 

    # Clean the extracted values to remove unwanted characters
    category = category.replace('(', '').replace(')', '').strip()
    question = question.replace('(', '').replace(')', '').strip()
    answer = answer.replace('(', '').replace(')', '').replace('"', '').strip()

    # Extract entities from the answer text
    answer_tokens = split_entities(answer, stop_words)
    answers_list = []
    for a0 in answer_tokens:
        # Find the corresponding Wikidata entity for each answer token
        ans = guess_wiki_entity(a0, sentence_embedding, embedder, topk=args.max_relevance)
        if isinstance(ans, list): answers_list.extend(ans)
        else: answers_list.append(ans)
    
    # Remove None values and duplicate entities from the answer list
    answers_list = list(filter(None, answers_list))
    answers_list = remove_duplicate_entities(answers_list)
    
    # Process the question text to extract entities using the NLP model
    doc = nlp(f'{category}. {question}')
    entities = extract_entities(doc, stop_words)
    entities_list = []
    for e0 in entities:
        # Find the corresponding Wikidata entity for each extracted entity
        ans = guess_wiki_entity(e0, sentence_embedding, embedder, topk=args.max_relevance)
        if isinstance(ans, list): entities_list.extend(ans)
        else: entities_list.append(ans)
            
    # Remove None values and duplicate entities from the entity list
    entities_list = list(filter(None, entities_list))
    entities_list = remove_duplicate_entities(entities_list)

    return idx, answers_list, entities_list

# Wrapper function to handle retry fetch in threads
def process_row_with_retry(row: Tuple[int, dict], nlp, embedder, args,
                           stop_words: Set[str] = set()) -> Tuple[int, List[dict], List[dict]]:
    # Retry the process_row function in case of temporary errors or timeouts
    return retry_fetch(process_row, row, nlp, embedder, args, stop_words, max_retries=3, timeout=10, verbose=True)

# Function to update the DataFrame with the results
def update_dataframe(df, results):
    for result in results:
        idx, answers_list, entities_list = result
        
        # Update the DataFrame with the extracted answer and question entities
        df.at[idx, 'Answer-QID'] = str([a0['QID'] for a0 in answers_list])
        df.at[idx, 'Question-QID'] = str([e0['QID'] for e0 in entities_list])
        
        df.at[idx, 'Answer-Entities'] = str([a0['Title'] for a0 in answers_list])
        df.at[idx, 'Question-Entities'] = str([e0['Title'] for e0 in entities_list])

if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    if args.stage_one:
        jeopardy_data = load_pandas(args.jeopardy_path) #[0:100]
    
        # Strip any extra spaces in the column names for safety
        jeopardy_data.columns = jeopardy_data.columns.str.strip()
    
        # Remove images and links from the questions
        jeopardy_data = jeopardy_data[~jeopardy_data['Question'].str.contains('href', case=False, na=False)]
    
        # If max_questions is specified, randomly sample a subset of questions
        if args.max_questions and args.max_questions < len(jeopardy_data):
            jeopardy_data = random_dataframes(jeopardy_data, args.max_questions, args.random_seed)
    
        # Download stopwords for entity extraction
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        
        # Define the device (use GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the tokenizer and model for Named Entity Recognition (NER)
        tokenizer = AutoTokenizer.from_pretrained(args.hh_ner_model)
        model = AutoModelForTokenClassification.from_pretrained(args.hh_ner_model).to(device)
        
        # Create a pipeline for NER with the specified aggregation strategy
        nlp = pipeline("ner", model=model, tokenizer=tokenizer,
                       aggregation_strategy=args.aggregation_strategy, device=0 if torch.cuda.is_available() else -1)
        
        # Load the embedding model (either OpenAI or Hugging Face based on the argument)
        if args.use_openai: embedding_gpt = OpenAIHandler(model=args.oa_embedding_model, encoding=args.oa_encoding_model)
        else: embedding_gpt = SentenceTransformer(args.hh_embedding_model)
        
        # Initialize new columns for storing processed data
        jeopardy_data['Question-Number'] = ['J-' + str(i0 + 1) for i0 in range(len(jeopardy_data))]
        jeopardy_data['Question-QID'] = [None] * len(jeopardy_data)
        jeopardy_data['Answer-QID'] = [None] * len(jeopardy_data)
        jeopardy_data['Question-Entities'] = [None] * len(jeopardy_data)
        jeopardy_data['Answer-Entities'] = [None] * len(jeopardy_data)
        
        # Reorder columns to have 'Question-Number' as the first column
        columns = ['Question-Number'] + [col for col in jeopardy_data.columns if col != 'Question-Number']
        jeopardy_data = jeopardy_data[columns]
        
        # Multi-threaded processing of rows for entity extraction
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers based on your CPU cores
            futures = [executor.submit(process_row_with_retry, row, nlp, embedding_gpt, args, stop_words) for row in jeopardy_data.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Jeopardy Data'):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing row: {e}")
        
        # Update the DataFrame with the results of entity extraction
        update_dataframe(jeopardy_data, results)
        
        # Save the unprocessed Jeopardy questions with extracted entities
        jeopardy_data.to_csv(args.jeopardy_unprocessed_path, index=False)
 
    #--------------------------------------------------------------------------
    if args.stage_two:
        # Load the data from the first stage
        jeopardy_data = load_pandas(args.jeopardy_unprocessed_path)
        
        # Filter out rows where either 'Answer-QID' or 'Question-QID' is empty
        filtered_jeopardy_data = jeopardy_data[
            (extract_literals(jeopardy_data['Answer-QID']).apply(lambda x: len(x) > 0 if isinstance(x, list) else False)) &  # Remove rows where 'Answer_QID' is None
            (extract_literals(jeopardy_data['Question-QID']).apply(lambda x: len(x) > 0 if isinstance(x, list) else False))  # Remove rows where 'Question_QID' is an empty list and has at least 2 QIDs
        ]
        
        # Save the final filtered questions with entities linked
        filtered_jeopardy_data.to_csv(args.jeopardy_processed_path, index=False)