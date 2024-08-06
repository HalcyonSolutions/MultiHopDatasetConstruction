# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:07:22 2024

@author: Eduin Hernandez

Summary: Webscrapes wikidata for the inverse relationships for a given relation.
"""

import pandas as pd
from tqdm import tqdm

from typing import Dict

from utils.basic import load_json
from utils.wikidata import fetch_inverse_relation

def add_relations(original_data: Dict[str, any]) -> Dict[str, any]:
    data = []
    
    for rdf, rdf_title in tqdm(original_data.items(), desc='Fetching relationship'):
        inverse_relation, subproperty_relation = fetch_inverse_relation(rdf)
        
        inv_rdf, inv_title = next(iter(inverse_relation.items())) if inverse_relation else (None, None)
        sub_rdf, sub_title = next(iter(subproperty_relation.items())) if subproperty_relation else (None, None)

        #prevent reverse relationship to self
        if inv_rdf == rdf: inv_rdf, inv_title = (None, None)
        if sub_rdf == rdf: sub_rdf, sub_title = (None, None)
        
        data.append({
            'rdf': rdf,
            'rdf title': rdf_title,
            'inverse rdf': inv_rdf,
            'inverse title': inv_title,
            'subproperty rdf': sub_rdf,
            'subproperty title': sub_title,
            'invert': False,
            'switch': False
        })
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Example usage
    input_file_path = './data/unique_properties_full.json'
    output_file_path = './data/unique_properties_valid.csv'
    
    #-----------------------------------------------------------------------------
    relations = load_json(input_file_path)
    # Step 2: Add inverse relations to the sorted data
    final_data = add_relations(relations)
    
    # Save the updated DataFrame
    final_data.to_csv(output_file_path, index=False)
    # Print the results
    print(final_data)