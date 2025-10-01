# -*- coding: utf-8 -*-
"""
Created on 2025-04-26

@author: Eduin Hernandez

Summary: This is a sample script that extracts matching nodes, neighborhoods,
 and paths between nodes in a graph using SimpleGraph. Key features include
 node extraction, neighborhood analysis, path finding between nodes, and
 printing detailed visual representations of the paths.
"""

from utils.simple_graph import SimpleGraph
from utils.configs import global_configs

def print_paths(paths) -> None:
    for nodes, rels in paths:
        hop_size = len(nodes) - 1
        parts = [f"Hop size {hop_size}: {nodes[0].get('Title', '?')}"]
        
        for i in range(1, len(nodes)):
            rel = rels[i - 1]
            title = rel.get('title') or rel.get('Title', '?')
            direction = rel.get('direction')
            arrow = f" --[{title}]--> " if direction == '->' else (
                    f" <--[{title}]-- " if direction == '<-' else
                    f" --[{title}]-- ")
            parts.append(arrow + nodes[i].get('Title', '?'))
        
        print(''.join(parts))

def print_paths_title_only(paths) -> None:
    for node_titles, rels in paths:
        hop_size = len(node_titles) - 1
        parts = [f"Hop size {hop_size}: {node_titles[0]}"]
        
        for i in range(1, len(node_titles)):
            rel_title, direction = rels[i - 1]
            arrow = f" --[{rel_title}]--> " if direction == '->' else (
                    f" <--[{rel_title}]-- " if direction == '<-' else
                    f" --[{rel_title}]-- ")
            parts.append(arrow + node_titles[i])
        
        print(''.join(parts))
    
if __name__ == '__main__':
    
    configs = global_configs('./configs/config_neo4j.ini')
    neo4j_parameters = configs['Neo4j']
    
    g = SimpleGraph(neo4j_parameters['uri'], neo4j_parameters['user'], neo4j_parameters['password'], database = 'kinshiphinton')
    
    #--------------------------------------------------------------------------
    'Matching Nodes and Neighborhoods'
    
    node = 'Christopher'                             # human
    
    nodeA = g.match_node(node)                      # extracting node info
    
    nodesB, relsB = g.match_related_nodes(node, direction='any', title_only=True) # extracting the neighborhood of parent node
    
    nodesC, relsC = g.match_related_nodes(node, direction='->', title_only=True)  # extracting nodes to which this one points to
    
    nodesD, relsD = g.match_related_nodes(node, direction='<-', title_only=True)   # extracting nodes to which point to this one
    
    print('Node:', nodeA['Title'])
    print('Related nodes:', nodesB)
    print('Related nodes (out):', nodesC)
    print('Related nodes (in):', nodesD)

    #--------------------------------------------------------------------------
    'Paths'
    
    start_node = 'Christopher'                   # grandfather   
    end_node   = 'Colin'                       # grandson
    min_hops = 2
    max_hops = 3
    limit = 5
    
    paths = g.find_path(start_node, end_node, min_hops=min_hops, max_hops=max_hops, limit=limit)
    print('Non-filtered paths:')
    print_paths(paths)

    rels = ['father', 'mother']
    paths = g.find_path(start_node, end_node, min_hops=min_hops, max_hops=max_hops, limit=limit, relationship_types=rels, title_only=True)
    print('Filtered paths:')
    print_paths_title_only(paths)
    
    # --------------------------------------------------------------------------
    'Neighboorhood'
    
    nodes = ['Charles', 'Jennifer']
    max_degree = 1
    limit = 0
    neighborhood = g.find_neighborhood(title_list=nodes, max_degree=max_degree, limit=limit, title_only=True)
    print(f'Neighborhood of {nodes}:\n{neighborhood}')

    #--------------------------------------------------------------------------
    'Relationships between Nodes'
    
    start_node = 'Christopher'
    end_node   = 'Arthur'
    
    rels = g.find_relationships(start_node, end_node)
    print(f'Relationships between {start_node} and {end_node}:\n{rels}')