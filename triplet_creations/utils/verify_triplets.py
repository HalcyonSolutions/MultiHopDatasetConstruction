# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:33:04 2024

@author: Eduin Hernandez

Summary:
This script provides a set of utility functions for handling and analyzing triplets
 and paths within QID-like data structures. The functions focus on mapping QID
 values to titles, confirming the existence of triplets, determining answerability
 of triplets, and filtering and visualizing paths based on specific criteria.
"""
import pandas as pd

from typing import Tuple, List

#------------------------------------------------------------------------------
'Mapping and Confirmation Functions'

def map_triplet_titles(tripled_df: pd.DataFrame, relation_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps QID and Property values in the triplet DataFrame to their corresponding titles based on given relation and node data.

    Args:
        tripled_df (pd.DataFrame): DataFrame containing triplets with 'head', 'relation', and 'tail' columns.
        relation_df (pd.DataFrame): DataFrame containing mapping of relation QID values to titles.
        node_df (pd.DataFrame): DataFrame containing mapping of node QID values to titles.

    Returns:
        pd.DataFrame: Updated DataFrame where QID values have been replaced with their corresponding titles.
    """
    qid_to_title_map = node_df.copy()
    qid_to_title_map = qid_to_title_map.set_index('QID')['Title']
    
    prop_to_title_map = relation_df.set_index('Property')['Title']
    tripled_df[['head', 'tail']] = tripled_df[['head', 'tail']].apply(lambda col: col.map(qid_to_title_map).fillna(col))
    tripled_df[['relation']] = tripled_df[['relation']].apply(lambda col: col.map(prop_to_title_map).fillna(col))
    return tripled_df

def confirm_triplets(df: pd.DataFrame, triplet_set: set, triplet_list: List[List[str]]) -> List[List[str]]:
    """
    Verifies if the triplets in the DataFrame exist in the provided triplet set or list.

    Args:
        df (pd.DataFrame): DataFrame containing triplets with 'head', 'relation', and 'tail' columns.
        triplet_set (set): Set of known triplets for quick lookup.
        triplet_list (List[List[str]]): List of known triplets for comparison.

    Returns:
        List[List[str]]: List of confirmed triplets that exist in the triplet set or match in the triplet list.
    """
    confirmed = []
    for _, row in df.iterrows():
        triplet = (row['head'], row['relation'], row['tail'])
        reverse_triplet = (row['tail'], row['relation'], row['head'])
        if 'Unknown' in [row['head'], row['tail']]:
            unknown_tail = row['tail'] == 'Unknown'
            partial_triplet = [row['head'], row['relation']] if unknown_tail else [row['tail'], row['relation']]
            matching_triplet = [t for t in triplet_list if t[:2] == partial_triplet] if unknown_tail else [t for t in triplet_list if [t[2], t[1]] == partial_triplet]
            matching_triplet_reverse = [t for t in triplet_list if [t[2], t[1]] == partial_triplet] if unknown_tail else [t for t in triplet_list if t[:2] == partial_triplet]
            if matching_triplet:
                confirmed.append([row['head'], row['relation'], row['tail']])
            if matching_triplet_reverse:
                confirmed.append([row['tail'], row['relation'], row['head']])
        elif triplet in triplet_set:
            confirmed.append([row['head'], row['relation'], row['tail']])
        elif reverse_triplet in triplet_set:
            confirmed.append([row['tail'], row['relation'], row['head']])
    
    # Remove duplicate triplets
    return [list(t) for t in set(tuple(t) for t in confirmed)]

#------------------------------------------------------------------------------
'Answerability Check Functions'

def is_answerable(df: pd.DataFrame, triplet_df: pd.DataFrame, answer: str) -> bool:
    """
    Determines if the question represented by a DataFrame of triplets can be answered using existing triplets.

    Args:
        df (pd.DataFrame): DataFrame containing triplets with 'head', 'relation', and 'tail' columns.
        triplet_df (pd.DataFrame): DataFrame containing the complete set of known triplets.
        answer (str): The entity to replace 'Unknown' with in the triplets.

    Returns:
        bool: True if there exists a matching triplet in `triplet_df`, otherwise False.
    """
    unknown_df = df[
        df['head'].eq('Unknown') | df['tail'].eq('Unknown')
    ].copy()
    unknown_df[['head', 'tail']] = unknown_df[['head', 'tail']].replace('Unknown', answer)
    
    # Check if any triplet in unknown_df exists in triplet_df
    unknown_df['exists'] = unknown_df.apply(lambda x: tuple(x) in triplet_df[['head', 'relation', 'tail']].itertuples(index=False, name=None), axis=1)
    return bool(unknown_df['exists'].any())

#------------------------------------------------------------------------------
'Path Sorting and Filtering Functions'

def sort_path_by_node_match(path_list: List[Tuple[List[str], List[str]]], valid_names: List[str]) -> Tuple[List[Tuple[List[str], List[str]]], List[int], List[int]]:
    """
    Sorts the path based on how many unique node names match the valid_names, and secondarily by the number of nodes if matches are equal.

    Args:
        path_list (List[Tuple[List[Any], List[Any]]]): A list of tuples representing nodes and relationships.
        valid_names (List[str]): A list of valid node names for matching.

    Returns:
        Tuple[List[Tuple[List[Any], List[Any]]], List[int], List[int]]: Sorted tuples, match counts, and node counts.
    """
    sorted_paths = sorted(
        path_list,
        key=lambda x: (len(set(x[0]) & set(valid_names)), -len(x[0])),
        reverse=True
    )
    match_counts = [len(set(t[0]) & set(valid_names)) for t in sorted_paths]
    node_counts = [len(t[0]) for t in sorted_paths]
    return sorted_paths, match_counts, node_counts

def filter_tuples_by_node(path_list: List[Tuple[List[str], List[str]]], ans_node: str) -> List[Tuple[List[str], List[str]]]:
    """
    Filters the tuples to only keep those that contain the specified answer node.

    Args:
        path_list (List[Tuple[List[str], List[str]]]): A list of tuples representing nodes and relationships.
        ans_node (str): The answer node to filter by.

    Returns:
        List[Tuple[List[str], List[str]]]: Filtered list of tuples.
    """
    # Filter the tuples to only keep those that contain the specified ans_node in their nodes
    filtered_paths = [t for t in path_list if ans_node in t[0]]
    return filtered_paths

#------------------------------------------------------------------------------
'Visualization Function'
def visualize_path(path: Tuple[List[str], List[str]], node_data: pd.DataFrame, rel_data: pd.DataFrame) -> str:
    """
    Converts a path into a readable format for visualization.

    Args:
        path (Tuple[List[str], List[str]]): A tuple representing nodes and relationships.
        node_data (pd.DataFrame): DataFrame containing node information.
        rel_data (pd.DataFrame): DataFrame containing relationship information.

    Returns:
        str: A string representation of the path.
    """
    if path == []: return "NO PATH FOUND"
    nodes = path[0]
    rels = path[1]
    path_vis = 'Hop size of ' + str(len(nodes) - 1) + ': ' + node_data.loc[nodes[0]]['Title']
    for i0 in range(1, len(nodes)):
        path_vis += ' --' + rel_data.loc[rels[i0-1]]['Title'] + '-- ' + node_data.loc[nodes[i0]]['Title']
    return path_vis