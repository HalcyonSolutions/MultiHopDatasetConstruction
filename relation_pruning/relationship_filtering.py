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

def filter_relationships(relations, rel_descriptions):
    if len(relations) == 0:
        raise ValueError('Variable "relations" is empty! No relationships to filter!')
    
    answers = []
    for relation, descption in zip(relations, rel_descriptions):
        prompt = f"""
        You are tasked to classify whether or not a relationship is informative or non-informative. I will provide you with a relationship title, and its description.
        Analyze a relationship logically to determine whether it provides meaningful information or not.
        For example, if the relationship is 'SEX', the only connection it might create between two nodes is a gender connection, this relationship is non-informative
        and should be filtered out. For each relationship, output '0' if it is non-informative and '1' if it is informative.
        Here is a relationship: {relations}, and its description: {descption}. Your output should be a 0 or 1 corresponding to the input relationships.
        Ensure that you only output a single digit. Do not store in memory, opeare only on the current query.Output only single digit!
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = (response.choices[0].message.content)
        print(f'Answer is {answer}')
        answers.append(float(answer))

    return answers

def run_prune_head(df, row):    
    answer = filter_relationships(df['Title'], df['Description'])
    df['non_informative'] = answer 
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prune relationships in a CSV file.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', type=str, default=None, help='Path to the input CSV file!')
    parser.add_argument('--row', type=int, default=0, help='Select the row from which the pruning will start!')

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['non_informative'] = [None]*len(df)
    df = run_prune_head(df, args.row)
    df.to_csv('/mnt/c/work/projects/DaRealMultiHop/triplet_creations/data/relation_data_filtered.csv')