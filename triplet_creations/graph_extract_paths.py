# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:10:04 2024

@author: Eduin Hernandez
"""
import ast
import pandas as pd

from utils.fb_wiki_graph import FbWikiGraph
from utils.basic import load_pandas
from utils.configs import global_configs

from typing import Dict, List, Tuple

class NodeRelFilter():
    """
    Graph Used to create and update Freebase-Wikidata Hybrid in Neo4j
    """
    def __init__(self, rels_path: str, rels_filter_path: str, nodes_path: str) -> None:
        self.rels_df        = load_pandas(rels_path)
        self.rel_filter_df  = load_pandas(rels_filter_path)
        self.nodes_df       = load_pandas(nodes_path)

    def get_parents(self, node: List[str]) -> List[str]:
        row = self.nodes_df.loc[self.nodes_df['RDF'] == node]
        if row['has_category'].iloc[0]: return ast.literal_eval(row['Category'].iloc[0])
        else: return [node]

    def parent_filters(self, parents: List[str]) -> List[str]:
        # Initialize combined_row as a boolean series with False for all columns except 'RDF'
        combined_row = pd.Series(False, index=self.rel_filter_df.columns.drop('RDF'))
        
        # Iterate through the list `parents` and apply the OR operation across the relevant rows
        for rdf_value in parents:
            if rdf_value in self.rel_filter_df['RDF'].values:  # Check if the RDF value exists in the DataFrame
                rows = self.rel_filter_df.loc[self.rel_filter_df['RDF'] == rdf_value].drop(columns='RDF')
                combined_row |= rows.any(axis=0)  # Apply OR operation with each row
        
        # Extract the column names where the entry is True
        return combined_row.index[combined_row].tolist()
    
    def _rel2neo4j(self, rel_prop: List[str]) -> List[str]:
        df = pd.DataFrame(rel_prop, columns=['Property'])
        return pd.merge(df, self.rels_df, on='Property', how='left')['Neo4j'].tolist()
    
    def nodes_rel_filters(self, start_node, end_node) -> List[str]:
        p0 = nrfilter.get_parents(start_node)
        p1 = nrfilter.get_parents(end_node)
        rels = nrfilter.parent_filters(p0 + p1)
        return nrfilter._rel2neo4j(rels)

if __name__ == '__main__':
    'Input'
    relation_path = './data/relation_data.csv'
    filter_path = './data/relationship_for_categories.csv'
    nodes_path = './data/category_mapping.csv'
    
    'Output'
    output_path = './data/two_hop.txt'
    #--------------------------------------------------------------------------
    'Load data'
    nodes       = load_pandas(nodes_path)
    
    nrfilter = NodeRelFilter(relation_path, filter_path, nodes_path)
    #--------------------------------------------------------------------------
    nodes_list = list(nodes['RDF'])
    
    start_node = nodes_list[0]
    end_node = nodes_list[105]
    
    rels = nrfilter.nodes_rel_filters(start_node, end_node)
    #--------------------------------------------------------------------------
    configs = global_configs('./configs/configs.ini')
    neo4j_parameters = configs['Neo4j']
    
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'], neo4j_parameters['password'])
    
    #--------------------------------------------------------------------------
    
    path = g.find_path(start_node, end_node, min_hops=2, max_hops=4,
                       limit=5, relationship_types=rels, rdf_only=True)
    
    # #--------------------------------------------------------------------------
    # for i0 in range(0, len(nodes_list)):
    #     for i1 in range(i0+1, len(nodes_list)):
    #         rels = nrfilter.nodes_rel_filters(nodes_list[i0], nodes_list[i1])
    #         path = g.find_path(nodes_list[i0], nodes_list[i1], min_hops=2, max_hops=2,
    #                            limit=5, relationship_types=rels, rdf_only=True)
            