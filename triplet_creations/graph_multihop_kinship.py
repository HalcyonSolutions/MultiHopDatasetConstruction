"""

"""
import argparse
import ast
import pandas as pd
from utils.basic import load_triplets, load_pandas

from utils.simple_graph import SimpleGraph
from utils.configs import global_configs
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate paths for kinship Hinton dataset")
    
    # Input
    parser.add_argument('--triplets-path', type=str, default='./data/triplets_kinship_hinton.txt',
                        help='Path to the text file containing valid triplets of entities used for filtering.')
    parser.add_argument('--hop-size', type=int, default=3,
                        help='The number of hops to consider for path finding.')
    
    # Output
    parser.add_argument('--path-output-path', type=str, default='./data/paths_kinship_hinton_3hop.txt',
                        help='Path to save the unprocessed paths.')
    
    return parser.parse_args()

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

def print_paths_list(paths) -> None:
    for path in paths:
        print(f"{path}")

def build_path_list(paths):
    path_list = []
    for nodes, rels in paths:
        parts = [f"{nodes[0].get('Title', '?')}"]
        for i in range(1, len(nodes)):
            rel = rels[i - 1]
            parts.extend([rel.get('title') or rel.get('Title', '?'), nodes[i].get('Title', '?')])
        path_list.append(parts)
    return path_list

def filter_directed_paths(paths, direction):
    filtered_paths = []
    for path in paths:
        _, rels = path
        if all(rel.get('direction') == direction for rel in rels):
            filtered_paths.append(path)
    return filtered_paths

def reverse_paths(paths):
    reversed_paths = []
    for path in paths:
        nodes, rels = path
        reversed_nodes = nodes[::-1]
        reversed_rels = rels[::-1]
        for rel in reversed_rels:
            if rel.get('direction') == '->':
                rel['direction'] = '<-'
            elif rel.get('direction') == '<-':
                rel['direction'] = '->'
        reversed_paths.append((reversed_nodes, reversed_rels))
    return reversed_paths

if __name__ == "__main__":
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    'Load 1 Hop Kinship Hinton Data'
    triplet_df = load_triplets(args.triplets_path)

    entities = set(triplet_df['head']).union(set(triplet_df['tail']))
    relations = set(triplet_df['relation'])

    upwards_rel = set(['uncle', 'aunt', 'father','mother'])
    downwards_rel = set(['son', 'daughter', 'nephew', 'niece'])
    lateral_rel = set(['brother', 'sister', 'husband', 'wife'])

    faceup = list(upwards_rel.union(lateral_rel))
    facedown = list(downwards_rel.union(lateral_rel))
    #----------------------------------------------------------------------------
    'Load Neo4j Graphs'

    configs = global_configs('./configs/config_neo4j.ini')
    neo4j_parameters = configs['Neo4j']
    
    g = SimpleGraph(neo4j_parameters['uri'], neo4j_parameters['user'], neo4j_parameters['password'], database = 'kinshiphinton')
    #----------------------------------------------------------------------------
    entities_list = list(entities)
    final_paths = []
    with tqdm(total=len(entities_list), desc=f'Creating MultiHop of Size {args.hop_size} - Part I') as pbar:
        for i0 in range(len(entities_list)):
            node_A = entities_list[i0]
            for i1 in range(i0 + 1, len(entities_list)):
                node_B = entities_list[i1]
                
                # Find paths between the two nodes
                paths = g.find_path(node_A,
                                    node_B, 
                                    min_hops=args.hop_size, 
                                    max_hops=args.hop_size, 
                                    limit=None, 
                                    relationship_types=faceup, 
                                    can_cycle=False,
                                    title_only=False)
                # Print the paths
                if len(paths) == 0: continue

                filtered_paths = filter_directed_paths(paths, direction='->')
                reversed_paths = filter_directed_paths(paths, direction='<-')
                if len(filtered_paths) > 0: 
                    final_paths.extend(build_path_list(filtered_paths))
                if len(reversed_paths) > 0:
                    reversed_paths = reverse_paths(reversed_paths)
                    final_paths.extend(build_path_list(reversed_paths))
            
            pbar.update(1)
            print_paths_list(final_paths[-5:])
            print(f"\n\n\n")
    
    with tqdm(total=len(entities_list), desc=f'Creating MultiHop of Size {args.hop_size} - Part II') as pbar:
        for i0 in range(len(entities_list)):
            node_A = entities_list[i0]
            for i1 in range(i0 + 1, len(entities_list)):
                node_B = entities_list[i1]
                
                # Find paths between the two nodes
                paths = g.find_path(node_A,
                                    node_B, 
                                    min_hops=args.hop_size, 
                                    max_hops=args.hop_size, 
                                    limit=None, 
                                    relationship_types=facedown, 
                                    can_cycle=False,
                                    title_only=False)
                # Print the paths
                if len(paths) == 0: continue

                filtered_paths = filter_directed_paths(paths, direction='->')
                reversed_paths = filter_directed_paths(paths, direction='<-')
                if len(filtered_paths) > 0: 
                    final_paths.extend(build_path_list(filtered_paths))
                if len(reversed_paths) > 0:
                    reversed_paths = reverse_paths(reversed_paths)
                    final_paths.extend(build_path_list(reversed_paths))
            
            pbar.update(1)
            print_paths_list(final_paths[-5:])
            print(f"\n\n\n")
    
    #--------------------------------------------------------------------------
    # save paths to a txt file via pandas
    paths_df = pd.DataFrame(final_paths)
    paths_df.to_csv(args.path_output_path, sep='\t', index=False, header=False)
    #---------------------------------------------------------------------------

    # qa_df = qa_df[['Question-Number', 'Question', 'Answer', 'Hops', 'Query-Entity', 'Query-Relation', 'Answer-Entity',
    #                'Paths', 'SplitLabel']]