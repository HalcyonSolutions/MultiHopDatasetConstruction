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

def prune_head(head, relationships, head_desc=None, len_rels=None):
    if len_rels == None:
        raise ValueError('Length of the relationships is None')
    if head_desc == None: # takes only head, and relatinoships variables
        prompt = f"""I will provide a single head, and a list of relationships. Your goal is to assign 0 and 1 to each of the relationship if it belongs to head. Assign 1 if a relationship belongs to a head, if it does not belong to it assign 0. Treat each relationship in the list independently. Result should be a list relationships with their corresponding score (0 or 1), separated by a , (comma) delimiter that follows the relationships in order as I provided. For example: Universe --> country 0, place of birth 1, place of death 1, spouse 0. I provide the head: {head}. I provide the list of 147 relationships: {relationships}, each relationship is separated by a delimiter , (comma)"""
    else: # takes only head, head_descriptions, and relatinoships variables
        prompt = f"""Using logical reasoning and factual analysis, assess the validity of a specified relationships with a given head. You will receive three pieces of information: the head (an entity or concept), a description of the head, and a proposed list of relationships. Your task is to determine whether the list of relationships logically applies to the head based on the description provided.

Please respond in the following format: [head] => (list of relationships with validity score) [country: validity score, place of birth: validity score, ...]. Assign a validity score of 1 if the relationship correctly pertains to the head, and 0 if it does not. In total there should be 153 relationships per head.

Input data:

Head: {head}
Head description: {head_desc}
Relationships: {relationships}
Ensure that your response adheres strictly to the specified format for clarity and consistency."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = (response.choices[0].message.content)

    digits = re.findall(r'\d', answer)
    digits = [float(num) for num in digits] 

    if contains_digit(head):
        digits = digits[-len_rels:]
    return digits


def run_prune_head(input_csv, method, row, relationships, heads, descriptions=None, th=3, failed=None):
    still_failed = []
    for i in range(row, len(heads)):
        if failed is not None:
            if heads[i] not in failed:
                continue

        df = pd.read_csv(input_csv)
        print(f'Row {i} --> Head: {heads[i]}')

        done = False
        f_count = 0 # if a model fails to complete a row in 'th' attemps, skip it (skipped rows are written into failed_head.txt)
        while done == False and f_count < th:
            try:
                if method == 1:
                    answer = prune_head(re.sub(r'[^a-zA-Z0-9\s]', '', heads[i]), ", ".join(relationships), head_desc=None, len_rels = len(relationships))
                if method == 2:
                    answer = prune_head(head=re.sub(r'[^a-zA-Z0-9\s]', '', heads[i]), relationships=", ".join(relationships), head_desc=re.sub(r'[^a-zA-Z0-9\s]', '', descriptions[i]), len_rels = len(relationships))

                df.loc[df['Title'] == heads[i], relationships] = answer
                
                df.to_csv(f'{input_csv}', index=False)
                
                print(f'\t*File {input_csv} is saved!\n')
                done = True

            except Exception as e:
                print(f'\t-Error: {e}')
                print(f'\t-Failed for Head {heads[i]}, trying again...')
                f_count += 1

        if f_count == th:
            with open('failed_heads.txt', 'a') as file:
                file.write('\n' + heads[i])
            print(f'\t*Failed to process the Head {heads[i]}! Wrote the head in the failed_heads.txt!\n')
            if failed is not None:
                still_failed.append(heads[i])
        del df
        gc.collect()

    if failed is not None:
        with open('failed_heads.txt', 'w') as f:
            for head in still_failed:
                f.write('\n' + head)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prune relationships in a CSV file.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', type=str, default=None, help='Path to the input CSV file!')
    parser.add_argument('--row', type=int, default=0, help='Select the row from which the pruning will start!')
    parser.add_argument('--method', type=int, default=1, help=(
        "Choose a method:\n"
        "  * 1: Works with Heads, and Relationships\n"
        "  * 2: Works with Heads, Relationships, and Descriptions,\n"
        "        it also contains more relationships"
    ))
    parser.add_argument('--th', type=int, default=3, help='Threshold for openai')
    parser.add_argument('--run_failed', type=str2bool, default=False, help='Will load the failed head from the failed_heads.txt, and prune them!')

    args = parser.parse_args()


    df = pd.read_csv(args.input)

    relationships = df.columns.to_list()
    if 'Title' in relationships:
        heads = df['Title'].to_list()
        relationships.remove('Title')
    else:
        raise ValueError("Your dataset doesn't contain 'Title' column")
    
    if 'Description' in relationships:
        descriptions = df['Description'].to_list()
        relationships.remove('Description')
    else:
        descriptions = None

    if args.run_failed:
        failed = []
        with open('datasets/failed_heads.txt', 'r') as f:
            for line in f:
                failed.append(line.strip())
        run_prune_head(args.input, args.method, args.row, relationships, heads, descriptions, args.th, failed=failed)
    else:
        run_prune_head(args.input, args.method, args.row, relationships, heads, descriptions, args.th, failed=None)

