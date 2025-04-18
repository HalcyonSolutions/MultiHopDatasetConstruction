# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:15:57 2024

@author: Eduin Hernandez

Summary: Webscrapes Wikidata for the information for each entity. 
            Uses threads to speed up information retrieval.
"""

import argparse

from utils.basic import str2bool

from utils.wikidata_v2 import process_entity_data

def parse_args():
    """
    Parses command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments from the command-line.
    """
    parser = argparse.ArgumentParser(description="Webscrape entity data from Wikidata with multi-threading support.")
    
    # Input file argument
    parser.add_argument('--input-set-path', type=str, nargs='+',  default=['./data/nodes_fb_wiki.txt'],
                        help='Path to the input text file containing entity identifiers.')
    
    # Output file argument
    parser.add_argument('--output-csv-path', type=str, default='./data/node_data_fb_wiki.csv',
                        help='Path to save the output CSV file containing entity data.')

    # Optional argument for maximum number of threads
    parser.add_argument('--max-workers', type=int, default=20,
                        help='Maximum number of threads for fetching data. Defaults to 10.')
    
    # Optional argument for number of rows to read from input file
    parser.add_argument('--nrows', type=int, default=None,
                        help='Number of rows to read from the input file. Defaults to None (read all rows).')
    
    # Optional argument for the number of retries if an HTTP request fails
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for failed requests. Defaults to 3.')

    # Optional argument for timeout for each request
    parser.add_argument('--timeout', type=int, default=2,
                        help='Timeout in seconds for each request. Defaults to 2 seconds.')

    # Optional argument for verbosity
    parser.add_argument('--verbose', type=str2bool, default='True',
                        help='Flag to enable verbose output (e.g., print errors during processing).')

    # Optional argument for failed log path
    parser.add_argument('--failed-log-path', type=str, default='./data/failed_ent_log.txt',
                        help='Path to save a log of failed entity retrievals. Defaults to ./data/failed_ent_log.txt.')

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    process_entity_data(
        file_path=args.input_set_path, 
        output_file_path=args.output_csv_path,
        nrows=args.nrows,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        timeout=args.timeout,
        verbose=args.verbose,
        failed_log_path=args.failed_log_path
    )