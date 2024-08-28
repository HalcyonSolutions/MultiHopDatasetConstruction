# This code will filter out the non-informative relationships

import csv
import argparse
import os
import pandas as pd
import re
from tqdm import tqdm

from utils.openai_api import OpenAIHandler
from utils.base_functions import str2bool

def pass_arguments():
    parser = argparse.ArgumentParser(description='Path quality evaluation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', type=str, default=None, help='Path to the input CSV file.')
    parser.add_argument('--output', type=str, default=None, help='Path to the output CSV file.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model.')
    parser.add_argument('--hop', type=int, default=2, help='Number of hops.')
    parser.add_argument('--from_scratch', type=str2bool, default=False, help='Start from scratch.')

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    return args


# Evaluate the quality of the path by prompting the bot
def evaluate_path(bot, path, model, hop=2):
    if len(path) == 0:
        raise ValueError('Variable "path" is empty! No relationships to filter!')

    # define a prompt that will be used to query the bot    
    prompt = f"""You are given a {hop} hop path in the format: node -> relationship -> node -> relationship -> node and so on. Where the first node is a starting node, and last node is an end node.
    These nodes and relationships come from the Knowledge Graph.
    Your task is to evaluate this path and give it a score from 0 to 1 based on its logical consistency and reasonableness. 
    A higher score indicates that the path makes sense and is logically sound, while a lower score indicates that the path is less coherent or reasonable. 

    Path: {path}

    Please provide only a single decimal number as your response.
    """

    # query the bot
    result = bot.query(prompt)
    answer = result['answer']

    # convert the answer to a decimal number
    decimal_pattern = r'^-?\d+(\.\d+)?$'
    if re.match(decimal_pattern, answer):
        return float(answer), result
    else:
        raise ValueError('Answer is not a decimal number, please modify your query! Ensure that the answer is a decimal number between 0 and 1.')


# Extract the actual names of the nodes and relationships from a dataframe row
def extract_path(row):
    df_nodes = pd.read_csv('../triplet_creations/data/rdf_data.csv') # helps to find info by RDF
    df_relations = pd.read_csv('../triplet_creations/data/relation_data.csv') # helps to find info by Property 
    path=''
    for e in row:
        if e[0] == 'Q':
            node = df_nodes.loc[df_nodes['RDF'] == e]
            node_title = node['Title'].item()
            #node_description = node['Description'].item()
            path += f'node:{node_title}'
        else:
            relation = df_relations.loc[df_relations['Property'] == e]
            rel_title = relation['Title'].item()
            path += f' --> relation:{rel_title} --> '

    if path=='':
        raise ValueError('Variable `path` can`t be None!')

    return path


# Evaluate all paths in the dataset
def run_evaluate_path(bot, df, model, hop, last_row=0):
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
        total_input_tokens = 0
        total_output_tokens = 0
        total_input_cost = 0
        total_output_cost = 0
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Paths Evaluation"):
            # skip the rows that were already processed
            if index < last_row:
                continue
            row = row[1:]
            path = extract_path(row)
            answer, result = evaluate_path(bot, path, model, hop)

            total_input_tokens += result['input_tokens']
            total_output_tokens += result['output_tokens']

            total_input_cost += result['input_cost']
            total_output_cost += result['output_cost']

            writer.writerow([answer])

    print(f'Overall:\n\t{total_input_tokens} were processed\n\t{total_input_cost} USD were spent')
    print(f'Overall:\n\t{total_output_tokens} were processed\n\t{total_output_cost} USD were spent')

    return True


# Merge the backup file with the original file
def merge(df, backup):
    # add a new column to the original file, called 'quality_score'
    # take the values from the backup file under the columns "score" and add them to the original file
    df['quality_score'] = backup['score']

    # save the updated file, and delete the backup file
    df.to_csv(args.output, index=False)
    os.remove('scored_backup.csv')
    print('scored_backup.csv is removed!')
    print('Merging is done!\n')


if __name__ == "__main__":
    args = pass_arguments()

    # read the dataset containing all paths
    df = pd.read_csv(args.input)

    # in case if scored_backup.csv exists, remove it and run the evaluation again
    # otherwise, continue from the last row, but the backup file must exist
    if args.from_scratch:
        if os.path.exists('scored_backup.csv'):
            os.remove('scored_backup.csv')
    else:
        # count the number of rows in the back up file
        last_row = len(pd.read_csv('scored_backup.csv'))

    # check that the dataset consists of 2hop+1 columns 
    if df.columns.size != 2*args.hop+1:
        raise ValueError('The dataset should contain 2*hop+1 columns!')

    # start the OpenAI bot
    bot = OpenAIHandler(model=args.model)

    # start evaluation
    status = run_evaluate_path(bot, df, args.model, args.hop, last_row)

    # if successful, merge
    if status:
        backup = pd.read_csv('scored_backup.csv')
        
        # if backup file isn't the same length as the original file, raise an error
        if len(df) != len(backup):
            raise ValueError('The backup file is not the same length as the original file!')
        
        merge(df, backup)
