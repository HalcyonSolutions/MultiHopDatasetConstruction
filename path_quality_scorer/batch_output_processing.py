import json
import pandas as pd
import argparse
from tqdm import tqdm 
import os, sys
import csv
import re
import numpy as np

def pass_arguments():
    parser = argparse.ArgumentParser(description='Path quality evaluation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_dataset', type=str, default=None, help='All paths scores will be saved in this file.')
    parser.add_argument('--input_folder', type=str, default=None, help='Path to the result file.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model.')
    args = parser.parse_args()

    return args


# Extract the content from the json line
def extract_content(line): 
    content = json.loads(line)
    return content['response']['body']['choices'][0]['message']['content']


# Extract the content from the json file
def run_extract_content(json_lines, model):
    contents = []
    for line in tqdm(json_lines, desc="Extracting content"):
        content = extract_content(line)
        contents.append(content)

    return contents


# Merge all batch_result files into a single one
def merge_batch_results(filenames):
    json_lines = []
    for input_file in filenames:
        with open(input_file, 'r') as f:
            json_lines += f.readlines()

    return json_lines


if __name__ == '__main__':
    args = pass_arguments()
    
    # read the dataset and the results
    df = pd.read_csv(f'./data/multihop/{args.input_dataset}')

    filenames = os.listdir(f'./data/batch_output/{args.input_folder}/')
    json_lines = merge_batch_results(f'./data/batch_output/{args.input_folder}/{filename}' for filename in filenames)

    content = run_extract_content(json_lines, args.model)

    # add a new column to the dataframe
    df['content'] = content
    total_score = []
    for i, text in enumerate(df['content'].to_list()):
        textlines = text.split('\n')
        for j, t in enumerate(textlines):
            if j == 2:
                digits = re.findall(r'\d+\.\d+', t)
                mean = np.mean([float(d) for d in digits]).item()
                total_score.append(mean)

    df['total_score'] = total_score

    df.to_csv(f'data/multihop/evaluated_{args.input_dataset}', index=False)
    print(f'file data/multihop/evaluated_{args.input_dataset} was saved!')
