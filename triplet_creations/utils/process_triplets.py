# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:51:53 2024

@author: Eduin Hernandez
"""
import pandas as pd
import json

from tqdm import tqdm

from typing import List, Tuple, Dict

#------------------------------------------------------------------------------
'Sorting Functions'
def _sort_json_by_keys(json_data: Dict[str, any]) -> Dict[str, any]:
    """
    Sorts a JSON dictionary by keys.
    
    Args:
        json_data (Dict[str, any]): The JSON dictionary to sort.
    
    Returns:
        Dict[str, any]: The sorted JSON dictionary.
    """
    sorted_items = sorted(json_data.items(), key=lambda item: int(item[0][1:]))
    
    return dict(sorted_items)

#------------------------------------------------------------------------------
'Split Functions'
def split_triplets(file_path: str, train_path: str, test_path: str, valid_path: str, split_rate:float = 0.8) -> None:
    """
    Splits triplets from a file into training, testing, and validation sets.
    
    Args:
        file_path (str): The path to the file containing the triplets.
        train_path (str): The path to save the training set.
        test_path (str): The path to save the testing set.
        valid_path (str): The path to save the validation set.
        split_rate (float): The proportion of data to use for the training set. Must be between 0.0 and 1.0.
    
    Returns:
        None
    """
    
    assert ((split_rate > 0.0) and (split_rate < 1.0)), 'Error! The split rate must be between 0.0 and 1.0!'
    
    test_rate = (1 - split_rate)/2
    
    # Load the triplets into a DataFrame
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    # Shuffle the DataFrame
    triplets_df = triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split indices
    train_end = int(split_rate * len(triplets_df))
    test_end = train_end + int(test_rate * len(triplets_df))
    
    # Split the DataFrame
    train_df = triplets_df.iloc[:train_end]
    test_df = triplets_df.iloc[train_end:test_end]
    valid_df = triplets_df.iloc[test_end:]
    
    # Save the splits to their respective files
    train_df.to_csv(train_path, sep='\t', header=False, index=False)
    test_df.to_csv(test_path, sep='\t', header=False, index=False)
    valid_df.to_csv(valid_path, sep='\t', header=False, index=False)

#------------------------------------------------------------------------------
'Analysis Functions'
def _get_tails_for_relationship(df: pd.DataFrame, relationship: str) -> pd.DataFrame:
    """
    Filters the DataFrame for the given relationship and counts the occurrences of each tail.

    Args:
        df (pd.DataFrame): The DataFrame containing the triplets.
        relationship (str): The relationship to filter by.

    Returns:
        pd.DataFrame: A DataFrame with tails and their counts for the given relationship.
    """
    filtered_df = df[df['relation'] == relationship]
    tail_counts = filtered_df['tail'].value_counts().reset_index()
    tail_counts.columns = ['tail', 'count']
    
    return tail_counts

def extract_tail_occurrences_by_relationship(file_path: str, relationship: str) -> pd.DataFrame:
    """
    Loads triplets from a file, filters by a given relationship, and counts the occurrences of each tail.

    Args:
        file_path (str): The path to the file containing the triplets.
        relationship (str): The relationship to filter by.

    Returns:
        pd.DataFrame: A DataFrame with tails and their counts for the given relationship.
    """
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    return _get_tails_for_relationship(df, relationship)

def create_relationship_presence_df(df: pd.DataFrame, nodes: list, relationship: str) -> pd.DataFrame:
    """
    Creates a DataFrame indicating the presence of a relationship for each node.

    Args:
        df (pd.DataFrame): The DataFrame containing the triplets.
        nodes (np.ndarray): An array of nodes.
        relationship (str): The relationship to check.

    Returns:
        pd.DataFrame: A DataFrame with booleans indicating the presence of the relationship and the corresponding tail.
    """
    presence_data = []

    for node in tqdm(nodes, desc="Processing nodes"):
        filtered_df = df[(df['head'] == node) & (df['relation'] == relationship)]
        if not filtered_df.empty:
            tails = filtered_df['tail'].tolist()
            presence_data.append({'head': node, 'has_relationship': True, 'tail': tails})
        else:
            presence_data.append({'head': node, 'has_relationship': False, 'tail': []})

    return pd.DataFrame(presence_data)

def count_relationships(file_path: str) -> pd.DataFrame:
    """
    Counts occurrences of each relationship in the triplets file.
    
    Args:
        file_path (str): The path to the file containing the triplets.
    
    Returns:
        pd.DataFrame: A DataFrame containing the counts of each relationship.
    """
    # Load the filtered triplets into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    # Count the occurrences of each relationship
    relation_counts = df['relation'].value_counts().reset_index()
    relation_counts.columns = ['relation', 'count']

    return relation_counts
    
def count_head_tail(file_path: str) -> Tuple[pd.DataFrame, set, set]:
    """
    Loads triplets from a file and counts the occurrences of entities as heads and tails.
    
    Args:
        file_path (str): The path to the file containing the triplets.
    
    Returns:
        Tuple[pd.DataFrame, set, set]: A tuple containing a DataFrame of merged head and tail counts, 
                                        a set of unique head entities, and a set of unique tail entities.
    """
    # Load the triplets into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    # Count the occurrences in the 'head' column
    head_counts = df['head'].value_counts().reset_index()
    head_counts.columns = ['entity', 'head_count']

    # Count the occurrences in the 'tail' column
    tail_counts = df['tail'].value_counts().reset_index()
    tail_counts.columns = ['entity', 'tail_count']

    # Merge the head and tail counts on the entity
    merged_counts = pd.merge(head_counts, tail_counts, on='entity', how='outer').fillna(0)

    # Convert counts to integers
    merged_counts['head_count'] = merged_counts['head_count'].astype(int)
    merged_counts['tail_count'] = merged_counts['tail_count'].astype(int)
    merged_counts['total_count'] = merged_counts['head_count'] + merged_counts['tail_count']
    return merged_counts, set(head_counts['entity']), set(tail_counts['entity'])

#------------------------------------------------------------------------------
'Modification Functions'
def filter_head_tail(file_path: str, pruning_num: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filters entities with 0 head count and a tail count greater than or equal to a given threshold.
    
    Args:
        file_path (str): The path to the file containing the triplets.
        pruning_num (int): The minimum tail count to include an entity. Default is 10.
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the filtered counts DataFrame and a list of filtered entities.
    """
    merged_counts, heads, _ = count_head_tail(file_path)

    # Filter entities with 0 head count and tail count >= pruning_num
    filtered_counts = merged_counts[(merged_counts['head_count'] == 0) & (merged_counts['tail_count'] >= pruning_num)]
    
    # Create the new entity list
    entity_list = list(heads | set(filtered_counts['entity']))

    return entity_list

def filter_triplets(file_path: str, entity_list: List[str], output_file_path: str) -> None:
    """
    Filters triplets to keep only those with entities from a given list, removes duplicates, and saves the result.
    
    Args:
        file_path (str): The path to the file containing the triplets.
        entity_list (List[str]): The list of entities to keep.
        output_file_path (str): The path to save the filtered triplets.
    """
    
    # Load the triplets into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    # Filter the DataFrame to keep only the triplets with entities in the entity_list
    filtered_df = df[(df['tail'].isin(entity_list))]
    
    # Filter out triplets where both head and tail do not contain 'Q'
    filtered_df = filtered_df[filtered_df['head'].str.contains('Q') & filtered_df['tail'].str.contains('Q')]
    
    # Remove any duplicate triplets
    filtered_df = filtered_df.drop_duplicates()

    # Store the new triplets
    filtered_df.to_csv(output_file_path, sep='\t', header=False, index=False)

def process_inverse_triplets(triplet_file_path: str, relationship_mapping_path: str, output_file_path: str) -> None:
    """
    Modifies triplets based on invert and switch conditions from a relationship mapping file.
    
    Args:
        triplet_file_path (str): The path to the file containing the triplets.
        relationship_mapping_path (str): The path to the file containing the relationship mapping.
        output_file_path (str): The path to save the modified triplets.
    """
    
    # Load the triplets into a DataFrame
    triplets_df = pd.read_csv(triplet_file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    # Load the relationship mapping
    relationship_mapping_df = pd.read_csv(relationship_mapping_path)
    
    # Invert triplets where 'invert' is True
    invert_relations = relationship_mapping_df[relationship_mapping_df['invert'] == True] #['rdf'].tolist()
    for _, row in invert_relations.iterrows():
        rdf, rel = row[['rdf','inverse rdf']]
        valid = triplets_df['relation'] == rdf
        triplets_df.loc[valid, ['head', 'tail']] = triplets_df.loc[valid, ['tail', 'head']].values
        triplets_df.loc[valid, ['relation']] = rel
    
    # Switch triplets where 'switch' is True
    switch_relations = relationship_mapping_df[relationship_mapping_df['switch'] == True]
    switch_mapping = dict(zip(switch_relations['rdf'], switch_relations['subproperty rdf']))
    triplets_df['relation'] = triplets_df['relation'].replace(switch_mapping)
    
    # Remove any duplicate triplets
    triplets_df = triplets_df.drop_duplicates()
    
    # Filter out triplets where both head and tail do not contain 'Q'
    triplets_df = triplets_df[triplets_df['head'].str.contains('Q') & triplets_df['tail'].str.contains('Q')]
    
    # Create an order-independent identifier for triplets
    triplets_df['unique_id'] = triplets_df.apply(lambda row: tuple(sorted([row['head'], row['tail']])) + (row['relation'],), axis=1)

    # Remove duplicates based on the unique identifier
    triplets_df = triplets_df.drop_duplicates(subset='unique_id').drop(columns='unique_id')
    
    # Store the modified triplets
    triplets_df.to_csv(output_file_path, sep='\t', header=False, index=False)

def extract_and_save_rdf_titles(relationship_mapping_path: str, relationship_counts_df: pd.DataFrame, output_json_path: str) -> None:
    """
    Extracts RDF titles for each relationship and saves them to a JSON file.
    
    Args:
        relationship_mapping_path (str): The path to the file containing the relationship mapping.
        relationship_counts_df (pd.DataFrame): The DataFrame containing relationship counts.
        output_json_path (str): The path to save the RDF titles JSON file.
    """
    # Load the relationship mapping
    relationship_mapping_df = pd.read_csv(relationship_mapping_path)
    
    rdf_title_dict = {}
    
    for rdf in relationship_counts_df['relation']:
        title = None
        # Check if the rdf is in the 'rdf' column
        if rdf in relationship_mapping_df['rdf'].values:
            title = relationship_mapping_df.loc[relationship_mapping_df['rdf'] == rdf, 'rdf title'].values[0]
        # If not, check in the 'subproperty rdf' column
        elif rdf in relationship_mapping_df['subproperty rdf'].values:
            title = relationship_mapping_df.loc[relationship_mapping_df['subproperty rdf'] == rdf, 'subproperty title'].values[0]
        
        if title:
            rdf_title_dict[rdf] = title
    
    rdf_title_dict = _sort_json_by_keys(rdf_title_dict)
    # Save the dictionary to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(rdf_title_dict, json_file, indent=4)