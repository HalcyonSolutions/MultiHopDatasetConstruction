# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:39:04 2024

@author: Eduin Hernandez
Sample code to extract matching nodes, neighborhood, and paths
"""
from utils.fb_wiki_graph import FbWikiGraph
from utils.configs import global_configs

def print_paths(paths) -> None:
    for path in paths:
        nodes = path[0]
        rels = path[1]
        path_vis = 'Hop size of ' + str(len(nodes) - 1) + ': ' + nodes[0]['Title']
        for i0 in range(1, len(nodes)):
            path_vis += ' --' + rels[i0-1]['Title'] + '--> ' + nodes[i0]['Title']
        print(path_vis)

def print_paths_rdf(paths) -> None:
    for path in paths:
        nodes = path[0]
        rels = path[1]
        path_vis = 'Hop size of ' + str(len(nodes) - 1) + ': ' + nodes[0]
        for i0 in range(1, len(nodes)):
            path_vis += ' --' + rels[i0-1] + '--> ' + nodes[i0]
        print(path_vis)
    
if __name__ == '__main__':
    
    configs = global_configs('./configs/configs.ini')
    neo4j_parameters = configs['Neo4j']
    
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'], neo4j_parameters['password'])
    
    #--------------------------------------------------------------------------
    'Matching Nodes and Neighborhoods'
    
    node = 'Q76'                             # human
    
    nodeA = g.match_node(node)              # extracting node info
    
    nodesB, relsB = g.match_related_nodes(node, direction='any') # extracting the neighborhood of parent node
    
    nodesC, relsC = g.match_related_nodes(node, direction='->')  # extracting nodes to which this one points to
    
    nodesD, relsD = g.match_related_nodes(node, direction='<-', rdf_only=True)   # extracting nodes to which point to this one
    
    #--------------------------------------------------------------------------
    'Paths'
    
    start_node = 'Q17'                      # Japan
    end_node   = 'Q183'                     # Germany
    min_hops = 2
    max_hops = 3
    limit = 100
    
    paths = g.find_path(start_node, end_node, min_hops=min_hops, max_hops=max_hops, limit=limit)
    print_paths(paths)

    rels = ['participant_in', 'member_of', 'shares_borders_with']
    paths = g.find_path(start_node, end_node, min_hops=min_hops, max_hops=max_hops, limit=limit, relationship_types=rels, rdf_only=True)
    print_paths_rdf(paths)