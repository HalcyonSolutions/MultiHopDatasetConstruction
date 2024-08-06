# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:39:04 2024

@author: Eduin Hernandez
Sample code to extract matching nodes and neighborhood
"""
from utils.fb_wiki_graph import FbWikiGraph

def print_paths(paths) -> None:
    for path in paths:
        nodes = path[0]
        rels = path[1]
        path_vis = 'Hop size of ' + str(len(nodes) - 1) + ': ' + nodes[0]['Title']
        for i0 in range(1, len(nodes)):
            path_vis += ' --' + rels[i0-1]['Title'] + '--> ' + nodes[i0]['Title']
        print(path_vis)

if __name__ == '__main__':
    
    g = FbWikiGraph('bolt://localhost:7687', 'neo4j', '11082000')
    
    #--------------------------------------------------------------------------
    'Matching Nodes and Neighborhoods'
    
    node = 'Q5'                             # human
    
    nodeA = g.match_node(node)              # extracting node info
    
    nodesB, relsB = g.match_connected(node) # extracting the neighborhood of parent node
    
    nodesC, relsC = g.match_outwards(node)  # extracting nodes to which this one points to
    
    nodesD, relsD = g.match_inwards(node)   # extracting nodes to which point to this one
    
    #_-------------------------------------------------------------------------
    'Paths'
    
    start_node = 'Q17'                      # Japan
    end_node   = 'Q183'                     # Germany
    min_hops = 2
    max_hops = 3
    limit = 100
    
    paths = g.find_path(start_node, end_node, min_hops=min_hops, max_hops=max_hops, limit=limit)
    
    print_paths(paths)
