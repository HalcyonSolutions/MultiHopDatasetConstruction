# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:51:14 2024

@author: Eduin Hernandez

Summary: Webscrapes Wiki-Data to extract the relationship information. Run this before fbwiki_relation_process.py.
"""
import pandas as pd
from tqdm import tqdm
from typing import Dict

from utils.basic import load_json
from utils.wikidata import fetch_relation_details

def add_relations(original_data: Dict[str, str]) -> Dict:
    data = []
    
    for rdf, rdf_title in tqdm(original_data.items(), desc='Fetching relationship'):
        info = fetch_relation_details(rdf)
        
        data.append(info)
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    property_path = './data/unique_properties_valid.json'
    output_file_path = './data/unique_properties_info.csv'
    
    #-----------------------------------------------------------------------------
    relations = load_json(property_path)
    
    df = add_relations(relations)
    
    df.to_csv(output_file_path, index=False)
