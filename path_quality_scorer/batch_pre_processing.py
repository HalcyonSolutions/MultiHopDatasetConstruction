# Create batch files from the dataset of paths

import os, sys
from tqdm import tqdm
import argparse
import pandas as pd
import tiktoken
import json

from utils.openai_api import pricing_input

def pass_arguments():
    parser = argparse.ArgumentParser(description='Path quality evaluation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dataset', type=str, default=None, help='Path to the input CSV file.')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to the output batch file.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model.')
    parser.add_argument('--hop', type=int, default=2, help='Number of hops.')

    args = parser.parse_args()

    return args


def create_prompt(path, model, hop=2):
    info = {}

    if len(path) == 0:
        raise ValueError('Variable "path" is empty! No relationships to filter!')

    # define a prompt that will be used to query the bot    
    info['prompt'] = f"""
    You are given a {hop}-hop path in the following format: node and its description -> relationship -> node and its description -> relationship -> node and its description, and so on. The first node is the starting point, and the last node is the endpoint.
    These nodes and relationships are derived from a Knowledge Graph. Your task is to evaluate this path to determine how good of a trivia question can be created from it.
    
    Score the path across four criteria:
    1) UNIQUENESS: How unique is the path, score from 0.0 to 1.0? Could the relationship yield multiple possible entities? (e.g., "Obama -> Born in -> USA -> Country of origin -> Average Joe")
    2) PERTINENCE: How relevant does the question seem, score from 0.0 to 1.0? Would a human likely ask something that would yield this "reasoning path", or does it feel randomly generated?
    3) COMMONALITY: Does the path represent common knowledge, score from 0.0 to 1.0? (e.g., "Obama -> Born in -> USA")
    4) CORRECTNESS: Is the inference logically sound, score from 0.0 to 1.0? Do you deem it correct information?
    
    Path: {path}
    Please provide your response in this format: 
    Path: <starting node> - <ending node>

    uniqueness [score], pertinence [score], commonality [score], correctness [score].
    
    Explanation of each score: <explanation>
    """  

    # count the number of tokens in the prompt
    encoding = tiktoken.encoding_for_model(model)

    info['tokens'] = len(encoding.encode(info['prompt']))

    return info


def extract_path(row):
    df_nodes = pd.read_csv('../triplet_creations/data/rdf_data.csv') # helps to find info by RDF
    df_relations = pd.read_csv('../triplet_creations/data/relation_data.csv') # helps to find info by Property 
    path=''
    for e in row:
        if e[0] == 'Q':
            node = df_nodes.loc[df_nodes['RDF'] == e]
            node_title = node['Title'].item()
            node_description = node['Description'].item()
            path += f'node:{node_title} and node description:{node_description}'
        else:
            relation = df_relations.loc[df_relations['Property'] == e]
            rel_title = relation['Title'].item()
            path += f' --> relation:{rel_title} --> '

    if path=='':
        raise ValueError('Variable `path` can`t be None!')

    return path


# Create a batch file that OpenAI batch API will process
def create_batch_file(df, model, hop, output_folder, output_name):
    token_limit = 1000000 # OpenAI has a 2M tokens limit, I keep it 1M

    info = {}
    batch_number = 1
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Batch file(s) preparation"):
        # Prepare data to form a request
        custom_id = f'request_{index}'
        method = 'POST' # HTTP method
        url = "/v1/chat/completions" # a.k.a endpoint
        system_role = 'system'
        system_content = "You are a helpful assistant."
        user_role = 'user'
        max_tokens = 1000

        # Prepare user content
        path = extract_path(row)
        promp_info = create_prompt(path, model, hop)
        user_content = promp_info['prompt']

        # Create a file to store the output batches, if it exists, it will append to it
        output_path = f'./data/batch_input/{output_folder}/{output_name}_batch_{batch_number}.jsonl'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # JSON object to be written into the file
        batch_data = {
            "custom_id": custom_id,
            "method": method,
            "url": url,
            "body": {
                "model": model,
                "messages": [
                    {"role": system_role, "content": system_content},
                    {"role": user_role, "content": user_content}
                ],
                "max_tokens": max_tokens
            }
        } 

        # Write the JSON object to a .jsonl file
        with open(output_path, 'a', newline='') as batch_file:
            json.dump(batch_data, batch_file)
            batch_file.write('\n')

        # check if the output_path is already in the dictionary, and if its not add it
        # helps to keep track of the number of tokens in each file
        if output_path not in info:
            info[output_path] = 0
        info[output_path] += promp_info['tokens']

        # check if the number of tokens exceeds the limit
        if info[output_path] > token_limit:
            batch_number += 1

    # printout each file and its token number from the info dictionary
    total_tokens = 0
    for key, value in info.items():
        print(f'File: {key} has {value} tokens.')
        total_tokens += value
    print()

    # Display the cost of processing the batch(es) using the selected model
    cost = total_tokens/1000 * pricing_input[model] / 2
    print(f'Processing the batch(es) using {model} will cost ${cost}.\n')

if __name__ == "__main__":
    args = pass_arguments()

    if args.input_dataset.endswith('.csv'):
        # Split the filename and extension
        output_name, _ = os.path.splitext(args.input_dataset)
    else:
        raise ValueError('Invalid input file format. The input file must have a .csv extension.')

    df = pd.read_csv(f'data/multihop/{args.input_dataset}')

    # check that the dataset consists of 2*hop+1 columns 
    if df.columns.size != 2*args.hop+1:
        raise ValueError('The dataset should contain 2*hop+1 columns!')

    # create the batch file(s) that OpenAI batch API will process
    create_batch_file(df, args.model, args.hop, args.output_folder, output_name)
