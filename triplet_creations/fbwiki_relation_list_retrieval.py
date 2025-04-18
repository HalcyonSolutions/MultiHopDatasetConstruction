# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:15:50 2024

@author: Eduin Hernandez

Summary: Webscrapes Wikidata to extract the entire list of relationships,
        extracts their data, and forms triplets from the hierarchy the hold.
"""
import argparse

from utils.basic import str2bool
from utils.wikidata_v2 import process_properties_list, process_relationship_data, process_relationship_hierarchy   

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process Wikidata properties and relationship data.")
    
    # Input arguments
    parser.add_argument('--max_properties', type=int, default=12541, 
                        help='Maximum number of properties to scrape')
    parser.add_argument('--limit', type=int, default=500, 
                        help='Limit per request for properties')
    parser.add_argument('--max-workers', type=int, default=15, 
                        help='Number of workers for scraping')
    parser.add_argument('--scrape-list', type=str2bool, default='True', 
                        help='Whether to enable scraping of property list')
    parser.add_argument('--scrape-data', type=str2bool, default='True', 
                        help='Whether to enable scraping of relationship data')
    parser.add_argument('--create-hierarchy', type=str2bool, default='True', 
                        help='Whether to enable hierarchy triplet creation')
    
    parser.add_argument('--verbose-error', type=str2bool, default='True', 
                        help='Whether to display the errors')
    
    # Output arguments
    parser.add_argument('--property-list-path', type=str, default='./data/relationship_wiki.txt',
                        help='Path to save the scraped property list')
    parser.add_argument('--property-dataframe-path', type=str, default='./data/relation_data_wiki.csv',
                        help='Path to save the complete properties list as a CSV file')
    parser.add_argument('--property-hierarchy-path', type=str, default='./data/relationships_hierarchy.txt',
                        help='Path to save the complete relationship hierarchy as a TXT file')
    
    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.scrape_list: 
        process_properties_list(args.property_list_path, args.max_properties, args.limit, max_workers=args.max_workers, verbose=args.verbose_error)
        
    if args.scrape_data: 
        process_relationship_data(args.property_list_path, args.property_dataframe_path, max_workers=args.max_workers, verbose=args.verbose_error)
    
    if args.create_hierarchy:
        process_relationship_hierarchy(args.property_list_path, args.property_hierarchy_path, max_workers=args.max_workers, verbose=args.verbose_error)