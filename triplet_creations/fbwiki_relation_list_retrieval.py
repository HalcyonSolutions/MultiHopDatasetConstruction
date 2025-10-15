# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:15:50 2024

@author: Eduin Hernandez

Summary: Webscrapes Wikidata to extract the entire list of relationships,
        extracts their data, and forms triplets from the hierarchy the hold.
"""
import argparse

from utils.basic import load_to_set, save_set_pandas
from utils.wikidata_v2 import process_properties_list, process_relationship_data, process_relationship_hierarchy   

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process Wikidata properties and relationship data.")
    
    # Input arguments
    parser.add_argument('--max_properties', type=int, default=12142, 
                        help='Maximum number of properties to scrape')
    parser.add_argument('--page-limit', type=int, default=500, 
                        help='Limit per request for properties')
    parser.add_argument('--max-workers', type=int, default=5, 
                        help='Number of workers for scraping')
    parser.add_argument('--max-rows', type=int, default=None, 
                        help='Number of rows to read from the entity list. Default is None, which means all rows.')
    parser.add_argument('--scrape-list', action='store_true',
                        help='Whether to enable scraping of property list')
    parser.add_argument('--scrape-data', action='store_true',
                        help='Whether to enable scraping of relationship data')
    parser.add_argument('--create-hierarchy', action='store_true',
                        help='Whether to enable hierarchy triplet creation')
    
    parser.add_argument('--verbose-error', action='store_true',
                        help='Whether to display the errors')
    
    # Output arguments
    parser.add_argument('--property-list-path', type=str, default='./data/vocabs/relationship_wiki.txt',
                        help='Path to save the scraped property list')
    parser.add_argument('--property-dataframe-path', type=str, default='./data/metadata/relation_data_wiki.csv',
                        help='Path to save the complete properties list as a CSV file')
    parser.add_argument('--property-hierarchy-path', type=str, default='./data/mappings/relationships_hierarchy.txt',
                        help='Path to save the complete relationship hierarchy as a TXT file')
    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    assert args.scrape_list or args.scrape_data or args.create_hierarchy, "Error: At least one of --scrape-list, --scrape-data, or --create-hierarchy must be set to True."

    if args.scrape_list: 
        df = process_properties_list(
            max_properties=args.max_properties, 
            page_limit=args.page_limit, 
            max_workers=args.max_workers, 
            verbose=args.verbose_error
        )

        save_set_pandas(df, args.property_list_path)
        print("\nData processed and saved to", args.property_list_path)
        
    if args.scrape_data:
        prop_list = list(load_to_set(args.property_list_path))[:args.max_rows]

        df = process_relationship_data(
            rel_list=prop_list,
            output_file_path=args.property_dataframe_path, 
            max_workers=args.max_workers, 
            verbose=args.verbose_error
        )

        df.to_csv(args.property_dataframe_path, index=False)
        print("\nRelationship data extracted and saved to", args.property_dataframe_path)
    
    if args.create_hierarchy:
        prop_list = list(load_to_set(args.property_list_path))[:args.max_rows]

        process_relationship_hierarchy(
            rel_list=prop_list, 
            output_file_path=args.property_hierarchy_path, 
            max_workers=args.max_workers, 
            verbose=args.verbose_error
        )