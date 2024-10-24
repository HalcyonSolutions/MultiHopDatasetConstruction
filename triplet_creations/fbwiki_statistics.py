# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:33:20 2024

@author: Eduin Hernandez

Summary: Calculates and visualize various statistics related to the FB-Wiki dataset,
         which is structured as a collection of triplets (head-relation-tail)
"""

import argparse

from utils.basic import str2bool
from utils.statistics_triplets import TripletsStats

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Calculates the Statistics of the Dataset")
    
    # Input arguments FB-Wiki-V3
    parser.add_argument('--entity-list-path', type=str, default='./data/nodes_fb_wiki.txt',
                        help='Path to the list of entities.')
    parser.add_argument('--entity-data-path', type=str, default='./data/node_data_fb_wiki.csv',
                        help='Path to the data of the entities.')
    parser.add_argument('--relationship-data-path', type=str, default='./data/relation_data_wiki.csv',
                        help='Path to the data of the relationship.')
    parser.add_argument('--triplets-data-path', type=str, default='./data/triplets_fb_wiki.txt',
                        help='Path to the relationship between entities.')
    
    
    parser.add_argument('--additional-statistics', type=str2bool, default='True',
                        help='Whether or not to show the additional statics, the ones that require more computation time.')
    parser.add_argument('--plot', type=str2bool, default='True',
                        help='Whether or not to plot the results')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    stats = TripletsStats(args.entity_list_path,
                  args.entity_data_path,
                  args.relationship_data_path,
                  args.triplets_data_path)
    
    #--------------------------------------------------------------------------
    'Basics '
    
    stats.basic_stats(True)
    
    stats.find_isolated_nodes(True)
    
    stats.calculate_degree_distribution(True)
    
    stats.calculate_graph_density(True)
    
    #--------------------------------------------------------------------------
    'Statisitics'
    node_freq, rel_freq = stats.calculate_triplet_frequency()
    
    relationship_count_per_node = stats.count_relationships_per_node()
    
    # #--------------------------------------------------------------------------
    if args.additional_statistics:
        'Additional Statistics'
        stats.calculate_clustering_coefficient(True)
        
        eigenvector_stats = stats.calculate_eigenvector_centrality()
    # -------------------------------------------------------------------------    
    if args.plot:
        'Plotting'
        
        # Frequency
        stats.plot_relationship_statistics(rel_freq, end_idx=60)
        stats.plot_node_statistics(node_freq, end_idx=60)
        
        # Unique Edges/Nodes
        stats.plot_node_diversity(relationship_count_per_node, end_idx=60)
        stats.plot_relationship_diversity(relationship_count_per_node, end_idx=60)
        
        # Eigenvector Centrality
        if args.additional_statistics: stats.plot_eigenvector_centrality(eigenvector_stats, end_idx=60)
        
        #TODO: Re-Implement Zip's Law
        # # Zip's Law
        # plot_zipfs_law(relation_stats['relation_count'].tolist(), 'Relations')
        # plot_zipfs_law(node_stats['total_count'].tolist(), 'Entities')
    
    
    