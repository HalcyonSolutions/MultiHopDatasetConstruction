# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:33:20 2024

@author: Eduin Hernandez

Summary: Calculates and visualize various statistics related to the FB-Wiki dataset,
         which is structured as a collection of triplets (head-relation-tail)
"""
import pandas as pd

from utils.basic import load_pandas
from utils.statistics_triplet import calculate_statistics, count_relationships_per_node
from utils.statistics_triplet import map_nodes_to_categories
from utils.statistics_triplet import plot_node_statistics, plot_distinct_nodes
from utils.statistics_triplet import plot_relationship_statistics, plot_distinct_edges
from utils.statistics_triplet import plot_category_node_statistics, plot_category_relationship_statistics
from utils.statistics_triplet import plot_zipfs_law

if __name__ == '__main__':
    'Input Data'
    relationship_data_path = './data/relation_data.csv'
    # node_data_path = './data/rdf_data.csv'
    node_data_path = './data/category_mapping.csv'
    triplet_data_path = './data/modified_triplet.txt'
    
    'Output Data'
    node_statistics_path = './data/node_statistics.csv'
    relationship_statistics_path = './data/relation_statistics.csv'
    relationship_count_per_node_path = './data/relation_per_node.csv'
    relationship_count_per_category_path = './data/relation_per_category.csv'
    #--------------------------------------------------------------------------
    'Loading Data'
    rels = load_pandas(relationship_data_path)
    
    nodes = load_pandas(node_data_path)

    triplet = pd.read_csv(triplet_data_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    #--------------------------------------------------------------------------
    'Calculating Statistics'
    node_stats, relation_stats = calculate_statistics(triplet)
    
    relationship_count_per_node = count_relationships_per_node(triplet, nodes, rels)
    
    relationship_count_per_category = map_nodes_to_categories(relationship_count_per_node, nodes)
    
    # -------------------------------------------------------------------------    
    'Plotting'
    
    # Frequency
    plot_relationship_statistics(relation_stats, rels, end_idx=60)
    plot_node_statistics(node_stats, nodes, end_idx=60)
    
    plot_category_node_statistics(relationship_count_per_category, nodes, end_idx=60)
    plot_category_relationship_statistics(relationship_count_per_category, rels, end_idx=60)
    
    # Unique Edges/Nodes
    plot_distinct_edges(relationship_count_per_node, nodes, end_idx=60)
    plot_distinct_nodes(relationship_count_per_node, rels, end_idx=60)
    
    plot_distinct_edges(relationship_count_per_category, nodes, end_idx=60)
    plot_distinct_nodes(relationship_count_per_category, rels, end_idx=60)
    
    # Zip's Law
    plot_zipfs_law(relation_stats['relation_count'].tolist(), 'Relations')
    plot_zipfs_law(node_stats['total_count'].tolist(), 'Entities')
    # -------------------------------------------------------------------------
    'Saving'
    
    node_stats.to_csv(node_statistics_path, index=False)
    relation_stats.to_csv(relationship_statistics_path, index=False)
    relationship_count_per_node.to_csv(relationship_count_per_node_path, index=False)
    relationship_count_per_category.to_csv(relationship_count_per_category_path, index=False)