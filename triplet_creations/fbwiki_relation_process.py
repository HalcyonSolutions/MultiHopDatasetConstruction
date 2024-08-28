# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:08:09 2024

@author: Eduin Hernandez
"""
from utils.basic import load_json, load_pandas
import regex as re
import pandas as pd

if __name__ == '__main__':
    'Input Files'
    property_path = './data/unique_properties_info.csv'
    clarification_path = './data/relationship_clarification.csv'
    head_rel_filter = './data/Head3K.csv'
    noninform_path = './data/relation_noninformative.csv'
    node_path = './data/categories_list.csv'
    
    
    'Output Path'
    output_path = './data/relation_data.csv'
    output_path2 = './data/relationship_for_categories.csv'
    
    #--------------------------------------------------------------------------
    property_map = load_pandas(property_path)
    clarification_map = load_pandas(clarification_path)
    noninform_map = load_pandas(noninform_path)
    
    clarification_map.rename(columns={'Old Categories': 'Title'}, inplace=True)
    clarification_map.rename(columns={'New Categories': 'Clarification Title'}, inplace=True)
    
    noninform_map.rename(columns={'non_informative': 'Non-Informative'}, inplace=True)
    noninform_map['Non-Informative'] = noninform_map['Non-Informative'].astype(bool)
    #--------------------------------------------------------------------------
    'Create a File that contains the relationship title, clarification title, and special property number'
    assert property_map['Title'].equals(clarification_map['Title']), "The 'Title' columns do not match exactly"
    assert property_map['Title'].equals(noninform_map['Title']), "The 'Title' columns do not match exactly"
    
    relation_mapping = pd.merge(property_map, clarification_map, on='Title')
    relation_mapping = pd.merge(relation_mapping, noninform_map[['Property', 'Non-Informative']], on='Property')
    relation_mapping['Neo4j'] = relation_mapping['Title'].apply(lambda x: re.sub('\W+', '_', x))
    
    relation_mapping.to_csv(output_path, header=True, index=False)
    
    #--------------------------------------------------------------------------
    'Re-processes the Relationship Pruning File for the Categories'
    
    head_rel = load_pandas(head_rel_filter)
    nodes = load_pandas(node_path)
    
    assert head_rel['Title'].equals(nodes['Title']), 'Error! Titles do not match on index values'
    
    head_rel2 = pd.merge(head_rel, nodes[['RDF']], left_index=True, right_index=True)
    head_rel2 = head_rel2.drop(['Title', 'Description'], axis=1)

    cols = list(head_rel2.columns)
    cols.insert(0, cols.pop(cols.index('RDF')))
    head_rel2 = head_rel2[cols]
    
    cols = list(head_rel2.columns)
    cols.pop(cols.index('RDF'))
    for c0 in cols:
        row = relation_mapping.loc[relation_mapping['Clarification Title'] == c0]

        head_rel2.rename(columns={c0: row['Property'].iloc[0]}, inplace=True)
    
    head_rel2.to_csv(output_path2, header=True, index=False)