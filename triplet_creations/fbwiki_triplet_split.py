# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:51:42 2024

@author: Eduin Hernandez


Summary: Splits the triplet dataset into training, testing, and validation sets for link prediction tasks.

This script allows the user to specify the paths for the input triplet file and the output paths for the train, test, 
and validation sets. It splits the triplet data into these subsets based on a specified split ratio.

Usage:
    The script can be run from the command line by providing the input triplet file and the output file paths for train, 
    test, and validation sets. A split ratio can also be provided to determine the proportion of the data that goes 
    into the training set, with the remaining data split equally between the test and validation sets.
"""

import argparse
import os
from utils.process_triplets import split_triplets, generate_dict

def parse_args():
    """
    Parses command-line arguments for the triplet splitting process.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Split a triplet dataset into training, testing, and validation sets.")
    
    # Input file
    parser.add_argument('--triplet-file-path', type=str, default='./data/link_prediction/Fb-Wiki/triplets.txt',
                        help='Path to the input triplet file to split.')
    
    # Output files
    parser.add_argument('--save-dir', type=str, default='./data/link_prediction/Fb-Wiki',
                        help='Directory to save the split datasets.')
    
    parser.add_argument('--generate-entity-dict', action='store_true',
                        help='Generate an entity dictionary from the triplet file.')
    
    parser.add_argument('--generate-relation-dict', action='store_true',
                        help='Generate a relation dictionary from the triplet file.')

    # Split ratio
    parser.add_argument('--split-rate', type=float, default=0.8,
                        help='Proportion of data to use for the training set (default is 0.8).')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Split the triplets into train, test, and validation sets
    split_triplets(
        args.triplet_file_path,
        os.path.join(args.save_dir, 'train.txt'),
        os.path.join(args.save_dir, 'test.txt'), 
        os.path.join(args.save_dir, 'valid.txt'), 
        split_rate=args.split_rate
    )

    if args.generate_entity_dict or args.generate_relation_dict:
        generate_dict(
            args.triplet_file_path,
            args.save_dir,
            save_entity = args.generate_entity_dict,
            save_relation = args.generate_relation_dict,
        )
    
    print("Triplets have been split into train, test, and valid files.")