import argparse
import numpy as np
import pandas as pd
import sys

def main(input_file, output_file):
    # Load the checker file
    try:
        checker_data = np.load('nodes_with_high_degree.npy', allow_pickle=True)
        checker_set = set(checker_data)  # Convert to set for faster lookups
    except Exception as e:
        print(f"Error loading checker file: {e}")
        return
    
    # loading the realationships that will be filtered
    tmp = pd.read_csv('relation_noninformative.csv')
    skip_rel = tmp[tmp['non_informative'] == 0]['Property'].to_list()

    # Open the input file and process it line by line
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Read the input file line by line
            for i, line in enumerate(infile):
                node1, node2, rel = line.strip().split('\t')
                if node1 not in checker_set and node2 not in checker_set:
                    if rel not in skip_rel:
                        outfile.write(f'{node1}\t{node2}\t{rel}\n')
                else:
                    print(f'row {i} is skipped')
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error reading input file: {e}")

if __name__ == "__main__":
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Check nodes against a checker file.")
    parser.add_argument('--input_file', type=str, help="Path to the input file.")
    parser.add_argument('--output_file', type=str, help='WDYM')

    args = parser.parse_args()
    main(args.input_file, args.output_file)

