import json
import pandas as pd
import argparse
from tqdm import tqdm 
import tiktoken

from utils.openai_api import pricing_input, pricing_output

def pass_arguments():
    parser = argparse.ArgumentParser(description='Path quality evaluation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', type=str, default=None, help='All paths scores will be saved in this file.')
    parser.add_argument('--results', type=str, default=None, help='Path to the result file.')
    parser.add_argument('--model', type=str, default=None, help='OpenAI model that you used to generate the results.')
    args = parser.parse_args()

    return args


# Extract the content from the json line
def extract_content(line): 
    content = json.loads(line)
    return content['response']['body']['choices'][0]['message']['content']


# Extract the content from the json file
def run_extract_content(json_lines, model):
    contents = []
    total_tokens = 0
    total_price = 0
    for line in tqdm(json_lines, desc="Extracting content"):
        content = extract_content(line)
        contents.append(content)

        # using tiktoken to check the token count
        encoding = tiktoken.encoding_for_model(model)
        tokens = len(encoding.encode(content))

        #print(f'Using model {model} was produced {tokens} tokens, which costs {tokens/1000 * pricing_output[model] / 2}')
        total_tokens += tokens
        total_price += tokens/1000 * pricing_output[model] / 2

    print(f'Total tokens: {total_tokens}')
    print(f'Total price: {total_price}')

    return contents


# Merge all batch_result files into a single one
def merge_batch_results(filenames):
    json_lines = []
    for filename in filenames:
        filename = filename.strip()
        with open(f'./data/batch_output/{filename}', 'r') as f:
            json_lines += f.readlines()

    return json_lines


if __name__ == '__main__':
    args = pass_arguments()
    
    # read the dataset and the results
    df = pd.read_csv(f'./data/multihop/{args.dataset}')

    # merge all batch_result files into a single one
    with open(f'./data/batch_output/{args.results}', 'r') as f:
        filenames = f.readlines()

    if len(filenames) == 1:
        with open(f"data/batch_output/{filenames[0].strip()}", 'r') as f:
            json_lines = f.readlines()
    elif len(filenames) == 0:
        raise ValueError('File is empty.')
    else:
        json_lines = merge_batch_results(filenames)

    results = run_extract_content(json_lines, args.model)
    
    # add a new column to the dataframe
    df['evaluation_score'] = results

    df.to_csv(f'data/multihop/{args.dataset}', index=False)
    print(f'file {args.dataset} was updated with the evaluation scores')
