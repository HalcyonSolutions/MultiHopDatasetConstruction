# This code will filter out the non-informative relationships

import csv
import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import gc
import sys
import re
import numpy as np
from tqdm import tqdm
import tiktoken

from utils.openai_models import OpenAIHandler

### Access OpenAI-API
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


### Functions
def contains_digit(text):
    return any(char.isdigit() for char in text)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def evaluate_path(path, model, hop=2):
    if len(path) == 0:
        raise ValueError('Variable "path" is empty! No relationships to filter!')
    
    # pricing per 1000 tokens
    pricing_input = {
        'gpt-4':0.03,
        'gpt-4-turbo':0.01,
        'gpt-4o-mini':0.00015
    }
    pricing_output = {
        'gpt-4':0.06,
        'gpt-4-turbo':0.03,
        'gpt-4o-mini':0.0006
    }

    # cl100k_base is a specific tokenization schema used by OpenAI models
    encoding = tiktoken.get_encoding("cl100k_base")

    encoding = tiktoken.encoding_for_model(model)

    prompt = f"""You are given a {hop} hop path in the format: node -> relationship -> node -> relationship -> node. Where the first node is a starting node, and last node is an end node.
    These nodes and relationships come from the Knowledge Graph.
    Your task is to evaluate this path and give it a score from 0 to 1 based on its logical consistency and reasonableness. 
    A higher score indicates that the path makes sense and is logically sound, while a lower score indicates that the path is less coherent or reasonable. 

    Path: {path}

    Please provide only a single decimal number as your response.
    """

    # record the number of input tokens,
    # later will be used to calculate the total number of tokens used,
    # as well as a cost
    input_tokens = len(encoding.encode(prompt))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = (response.choices[0].message.content)

    # get the total number of tokens used for this operation
    output_tokens = len(encoding.encode(answer))
    total_tokens = input_tokens + output_tokens
    #print(f'Total tokens used for this query -> {total_tokens}')

    # calculate the cost
    input_cost = input_tokens/1000 * pricing_input[model]
    output_cost = output_tokens/1000 * pricing_output[model]
    total_cost = input_cost + output_cost
    #print(f'Total USD$ used for this query -> {total_cost}')

    decimal_pattern = r'^-?\d+(\.\d+)?$'
    if re.match(decimal_pattern, answer):
        return float(answer), total_tokens, total_cost
    else:
        raise ValueError('Answer is not a decimal number, please modify your query!')

def extract_path(row):
    df_nodes = pd.read_csv('../triplet_creations/data/rdf_data.csv') # helps to find info by RDF
    df_relations = pd.read_csv('../triplet_creations/data/relation_data.csv') # helps to find info by Property 
    path=''
    for e in row:
        if e[0] == 'Q':
            node = df_nodes.loc[df_nodes['RDF'] == e]
            node_title = node['Title'].item()
            node_description = node['Description'].item()
            path += f'node:{node_title}'
        else:
            relation = df_relations.loc[df_relations['Property'] == e]
            rel_title = relation['Title'].item()
            path += f' --> relation:{rel_title} --> '

    if path=='':
        raise ValueError('Variable `path` can`t be None!')

    return path

def run_evaluate_path(df, model, hop):
    output_path = 'scored_backup.csv'
    if os.path.exists(output_path):
        exists = True
    else:
        exists = False

    # use backup file while performing operations, 
    # after `run_evaluate_path` returns True, 
    # backup file will be merged with the original file, and will be deleted
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # add name of the columns
        if not exists:
            writer.writerow(['row', 'score'])
        
        # process each row in the dataset, and write the results in the backup file
        total_tokens = 0
        total_cost = 0
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Paths Evaluation"):
            row = row[1:]
            path = extract_path(row)
            answer, used_tokens, spent_money = evaluate_path(path, model, hop)
            total_tokens += used_tokens
            total_cost += spent_money
            writer.writerow([answer])

    print(f'Overall:\n\t{total_tokens} were processed\n\t{total_cost} USD were spent')
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prune relationships in a CSV file.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', type=str, default=None, help='Path to the input CSV file.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model.')
    parser.add_argument('--hop', type=int, default=2, help='Number of hops.')

    args = parser.parse_args()
    
    # read the dataset containing all paths
    df = pd.read_csv(args.input)
    
    # start scoring
    run_evaluate_path(df, args.model, args.hop)

    #if run_evaluate_path(df, args.output):
        # merge

    print('Completed! \u2665') # \u2665, is a heart symbol, only visible if your font has this information
