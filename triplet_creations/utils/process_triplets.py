# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:51:53 2024

@author: Eduin Hernandez

Summary:
The `process_triplets` package provides tools for loading, processing, analyzing, and modifying triplet datasets,
typically used in knowledge graphs or relational datasets. It includes functions to manage inverse relationships, 
handle duplicates, and extract entities and relationships for analysis.

Core functionalities:

- **Processing Functions**: Extract and process inverse relationships, subproperties, and apply remapping.
- **Duplicate Handling**: Identify, remap, and remove duplicate relations, including reverse duplicates.
- **Analysis Functions**: Count entity and relationship occurrences, and collect entities based on relationships.
- **Extraction Functions**: Extract unique entities or relationships from triplet datasets.
- **Modification Functions**: Filter and modify triplets based on entity sets or specific criteria.
- **Split Functions**: Split triplet datasets into training, testing, and validation sets for model training.

This package is ideal for processing large-scale triplet datasets and preparing data for relational analysis or machine learning.
"""
import pandas as pd

from typing import List, Tuple, Dict, Set, Union

from utils.basic import load_triplets, load_to_dict, load_to_set, load_pandas
from utils.basic import save_triplets, save_set_pandas, save_dict_pandas
from utils.basic import sort_by_qid

#------------------------------------------------------------------------------
"""

~~~ Processing Functions ~~~

These functions are responsible for extracting relationships (inverse or subproperty),
processing inverse relations, and handling duplicate removal or remapping.

"""
def extract_inverses(triplet_path:str, hierarchy_path: str, include_self_inv: bool = False) -> pd.DataFrame:
    """
    Extracts inverse relationships from a triplet dataset based on a specified relationship hierarchy.
    
    Args:
        triplet_path (str): The path to the triplet file containing entity relationships.
        hierarchy_path (str): The path to the hierarchy file that includes relationship information.
        include_self_inv (bool, optional): Whether to include self-inverses (where 'head' and 'tail' are the same). Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the inverse relationships, with 'head' and 'tail' entities that have an inverse relation (P1696).
    """
    rel_set = set(load_triplets(triplet_path)['relation'].tolist())
    
    rel_hier_df = load_triplets(hierarchy_path)
    
    rel_inv = rel_hier_df[rel_hier_df['relation'].isin(['P1696'])]
    rel_inv = rel_inv[rel_inv['head'].isin(rel_set) & rel_inv['tail'].isin(rel_set)]
    rel_inv = sort_by_qid(rel_inv, column_name='head')
    
    # Remove self inverse
    if not(include_self_inv): rel_inv = rel_inv[rel_inv['head'] != rel_inv['tail']]
    
    return rel_inv

def extract_subproperties(hierarchy_path: str) -> pd.DataFrame:
    """
    Extracts subproperty relationships (P1647) from a hierarchy dataset.
    
    Args:
        hierarchy_path (str): The path to the hierarchy file that includes relationship information.
    
    Returns:
        pd.DataFrame: A DataFrame containing subproperty relationships (P1647) between entities.
    """
    rel_hier_df = load_triplets(hierarchy_path)
    
    return rel_hier_df[rel_hier_df['relation'].isin(['P1647'])]

def process_inverses_in_triplets(triplet_file_path: str, hierarchy_mapping: str, include_self_inv: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Processes inverses in triplets by extracting inverse relations, handling duplicates, applying remapping, and removing reverse duplicates.

    Args:
        triplet_file_path (str): The file path to the triplet dataset.
        hierarchy_mapping (str): The file path to the hierarchy mapping dataset.
        include_self_inv (bool, optional): Whether to include self-inverses (where 'head' and 'tail' are the same). Defaults to False.

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: A tuple containing the processed DataFrame of inverse relations and a dictionary of remappings.
    """
    rel_inv = extract_inverses(triplet_file_path, hierarchy_mapping, include_self_inv=include_self_inv)
    rel_subprop = extract_subproperties(hierarchy_mapping)
    
    rel_inv, remapping = process_inverse_relations(rel_inv, rel_subprop)
    
    return rel_inv, remapping

def process_inverse_relations(rel_inv: pd.DataFrame, rel_subprop: pd.DataFrame) -> pd.DataFrame:
    """
    Processes inverse relations by handling duplicates, applying remapping, and removing reverse duplicates.

    Args:
        rel_inv (pd.DataFrame): The DataFrame containing inverse relations.
        rel_subprop (pd.DataFrame): The DataFrame containing sub-property relations.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the processed DataFrame and a dictionary of remappings.
    """
    # Handle tail duplicates
    duplicate_tails = _count_duplicates(rel_inv, 'tail')
    remapping_tail, rows_to_drop_tail = _process_duplicate_inverse_relations(rel_inv, rel_subprop, duplicate_tails, 'tail')
    rel_inv = _apply_remapping_to_relations(rel_inv, remapping_tail, 'head')
    rel_inv = rel_inv.drop(list(rows_to_drop_tail))
    rel_inv = rel_inv.drop_duplicates()

    # Handle head duplicates
    duplicate_heads = _count_duplicates(rel_inv, 'head')
    remapping_head, rows_to_drop_head = _process_duplicate_inverse_relations(rel_inv, rel_subprop, duplicate_heads, 'head')
    rel_inv = _apply_remapping_to_relations(rel_inv, remapping_head, 'tail')
    rel_inv = rel_inv.drop(list(rows_to_drop_head))
    rel_inv = rel_inv.drop_duplicates()

    # Combine remappings (for reporting or further processing if needed)
    remapping = {}
    remapping.update(remapping_tail)
    remapping.update(remapping_head)

    # Remove reverse duplicates
    rel_inv = _remove_reverse_duplicates_in_relations(rel_inv)
    
    rel_inv = sort_by_qid(rel_inv, column_name='head')

    return rel_inv, remapping

#------------------------------------------------------------------------------
"""

~~~ Duplicate Handling Functions ~~~

These functions deal with identifying, remapping, and removing duplicate relationships.

"""

def _count_duplicates(df: pd.DataFrame, column: str) -> pd.Index:
    """
    Count the occurrences of values in a specific column that appear more than once.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to count duplicate values in.

    Returns:
        pd.Index: Index of values that appear more than once in the specified column.
    """
    counts = df[column].value_counts()
    return counts[counts > 1].index

def _process_duplicate_inverse_relations(df: pd.DataFrame, rel_subprop: pd.DataFrame, duplicate_values: pd.Index, column: str) -> (dict, set):
    """
    Processes duplicate inverse relations in the DataFrame by remapping or dropping rows based on sub-property relationships.

    Args:
        df (pd.DataFrame): The DataFrame containing inverse relations.
        rel_subprop (pd.DataFrame): The DataFrame containing sub-property relations.
        duplicate_values (pd.Index): The values that are duplicated in the specified column ('head' or 'tail').
        column (str): The column to process for duplicates ('tail' or 'head').

    Returns:
        Tuple[dict, set]: A tuple containing a dictionary for remapping relations and a set of row indices to drop.
    """
    remapping = {}
    rows_to_drop = set()
    
    for i0, row in df[df[column].isin(duplicate_values)].iterrows():
        # Determine the opposite column for sub-property check
        opposite_column = 'head' if column == 'tail' else 'tail'
        val_opposite_subprop = rel_subprop[rel_subprop['head'] == row[opposite_column]]
        
        # Find other rows with the same column value
        same_value_rows = df[df[column] == row[column]]
        same_value_rows = same_value_rows[same_value_rows.index != i0]  # Skip self
        
        for j0, other_row in same_value_rows.iterrows():
            # Check sub-property for the other opposite column value
            other_opposite_subprop = rel_subprop[rel_subprop['head'] == other_row[opposite_column]]
            
            if not val_opposite_subprop.empty and not other_opposite_subprop.empty:
                if val_opposite_subprop['tail'].iloc[0] == other_opposite_subprop['tail'].iloc[0]:
                    remapping[df.loc[i0, opposite_column]] = val_opposite_subprop['tail'].iloc[0]
                else:  # If sub-properties don't match, check inverse relation
                    inverse_row = df[(df['head'] == row[column]) & (df['tail'] == row[opposite_column])]
                    if inverse_row.empty: rows_to_drop.add(i0)
    
    return remapping, rows_to_drop


def _apply_remapping_to_relations(df: pd.DataFrame, remapping: dict, column: str) -> pd.DataFrame:
    """
    Applies remapping to a specific column in the relations DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the relations.
        remapping (dict): A dictionary containing the remapping values.
        column (str): The column to remap ('head' or 'tail').

    Returns:
        pd.DataFrame: The updated DataFrame after applying the remapping.
    """
    for i0, row in df[df[column].isin(remapping.keys())].iterrows():
        df.loc[i0, column] = remapping[row[column]]
    
    return df


def _remove_reverse_duplicates_in_relations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes reverse duplicates from the relations DataFrame by sorting 'head' and 'tail' values.

    Args:
        df (pd.DataFrame): The DataFrame containing the relations.

    Returns:
        pd.DataFrame: The DataFrame with reverse duplicates removed.
    """
    df['sorted_head'] = df[['head', 'tail']].apply(lambda x: tuple(sorted(x)), axis=1)
    df = df.drop_duplicates(subset=['sorted_head'])
    df = df.drop(columns=['sorted_head'])  # Clean up temporary column
    return df

#------------------------------------------------------------------------------
"""

~~~ Analysis Functions ~~~

These functions provide analysis of triplets, including counting occurrences and
extracting specific types of entities and relationships.

"""
def count_entity_occurance(file_path: Union[str, List[str]]) -> Tuple[pd.DataFrame, set, set]:
    """
    Loads triplets from a file and counts the occurrences of entities as heads and tails.
    
    Args:
        file_path (str or list): The path to the file containing the triplets.
    
    Returns:
        Tuple[pd.DataFrame, set, set]: A tuple containing a DataFrame of merged head and tail counts, 
                                        a set of unique head entities, and a set of unique tail entities.
    """
    
    df = load_triplets(file_path)

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

def count_relationship_occurance(file_path: Union[str, List[str]]) -> pd.DataFrame:
    """
    Counts occurrences of each relationship in the triplets file.
    
    Args:
        file_path (str or list): The path to the file containing the triplets.
    
    Returns:
        pd.DataFrame: A DataFrame containing the counts of each relationship.
    """
    # Load the filtered triplets into a DataFrame
    df = load_triplets(file_path)

    # Count the occurrences of each relationship
    relation_counts = df['relation'].value_counts().reset_index()
    relation_counts.columns = ['relation', 'count']

    return relation_counts

def collect_head_given_relation(file_path: Union[str, List[str]], relationships: List[str]) -> Set[str]:
    """
    Collects the unique head entities that are associated with specific relationships from a set of triplets.
    
    Args:
        file_path (Union[str, List[str]]): The path to the file or list of files containing the triplets.
        relationships (List[str]): A list of relationship types (relations) to filter the triplets by.
    
    Returns:
        Set[str]: A set of unique tail entities that are connected to the specified relationships.
    """

    df = load_triplets(file_path)
        
    entity_set = set()
    for r0 in relationships:
        df2 = df[df['relation'] == r0]
        entity_set.update(df2['head'].tolist())
    return entity_set

def collect_tails_given_relation(file_path: Union[str, List[str]], relationships: List[str]) -> Set[str]:
    """
    Collects the unique tail entities that are associated with specific relationships from a set of triplets.
    
    Args:
        file_path (Union[str, List[str]]): The path to the file or list of files containing the triplets.
        relationships (List[str]): A list of relationship types (relations) to filter the triplets by.
    
    Returns:
        Set[str]: A set of unique tail entities that are connected to the specified relationships.
    """

    df = load_triplets(file_path)
        
    entity_set = set()
    for r0 in relationships:
        df2 = df[df['relation'] == r0]
        entity_set.update(df2['tail'].tolist())
    return entity_set

def collect_entities_via_pruning(file_path: Union[str, List[str]], pruning_num: int = 10) -> Set[str]:
    """
    Collects the entities with 0 head count and a tail count greater than or equal to a given threshold.
    
    Args:
        file_path (str or list): The path to the file containing the triplets.
        pruning_num (int): The minimum tail count to include an entity. Default is 10.
    
    Returns:
        Set[str]: A Set containing the ID of filtered entities.
    """
    merged_counts, heads, _ = count_entity_occurance(file_path)

    # Filter entities with 0 head count and tail count >= pruning_num
    filtered_counts = merged_counts[(merged_counts['head_count'] == 0) & (merged_counts['tail_count'] >= pruning_num)]
    
    # Create the new entity list
    entity_set = set(heads | set(filtered_counts['entity']))

    return entity_set

def find_missing_entities(before_path: str, after_path: str) -> set:
    """
    Extracts the set of entities that are present in the triplets from the 'after' dataset but missing from the 'before' dataset.
    
    Args:
        before_path (str): The file path to the dataset containing the triplets before processing.
        after_path (str): The file path to the dataset containing the triplets after processing.
    
    Returns:
        set: A set of entities present in the 'after' dataset but missing in the 'before' dataset.
    """
    
    after_df = load_triplets(after_path)
    
    before_set = set(load_triplets(before_path)['head'].tolist())
    
    after_set = set(after_df['head'].tolist()) | set(after_df['tail'].tolist())
    
    return after_set - before_set

#------------------------------------------------------------------------------
"""

~~~ Modification Functions ~~~

These functions modify the triplet dataset, including filtering and remapping
based on processed data or entity sets.

"""
def correct_forwarding(file_path: str, entity_forwarding_path: str) -> str:
    """
    Corrects the triplet dataset by replacing entities with their forwarding counterparts based on a mapping file.

    Args:
        file_path (str): The path to the triplet dataset file.
        entity_forwarding_path (str): The path to the entity forwarding mapping file.

    Returns:
        str: The path to the corrected triplet dataset file.
    """
    forward_pd = load_pandas(entity_forwarding_path)

    # Create a mapping dictionary from forward_pd
    forward_mapping = dict(zip(forward_pd["QID-to"], forward_pd["QID-from"]))

    triplet_set = load_triplets(file_path)

    # Replace the values in triplet_set using the mapping
    triplet_set["head"] = triplet_set["head"].map(forward_mapping).fillna(triplet_set["head"])
    triplet_set["tail"] = triplet_set["tail"].map(forward_mapping).fillna(triplet_set["tail"])

    if type(file_path) == list: file_path = file_path[0]

    forward_triplet_path = file_path.replace('.txt', '_forwarded.txt')
    save_triplets(triplet_set, forward_triplet_path)

    return forward_triplet_path

def filter_triplets_by_entities(file_path: Union[str, List[str]], entity_list: Set[str], output_file_path: str) -> None:
    """
    Filters triplets to keep only those with entities from a given list, removes duplicates, and saves the result.
    
    Args:
        file_path (str or list): The path to the file containing the triplets.
        entity_list (Set[str]): The set of entities to keep.
        output_file_path (str): The path to save the filtered triplets.
    """
    
    df = load_triplets(file_path)

    # Filter the DataFrame to keep only the triplets with entities in the entity_list
    filtered_df = df[(df['tail'].isin(entity_list))]
    
    # Filter out triplets where both head and tail do not contain 'Q'
    filtered_df = filtered_df[filtered_df['head'].str.contains('Q') & filtered_df['tail'].str.contains('Q')]
    
    # Remove any duplicate triplets
    filtered_df = filtered_df.drop_duplicates()

    # Store the new triplets
    save_triplets(filtered_df, output_file_path)

def clean_triplet_relations(triplet_filtered_file_path: str, 
                            triplet_processed_file_path: str, 
                            relationship_hierarchy_mapping: str = None, 
                            inverse_mapping_path: str = None,
                            reverse_mapping_path: str = None,
                            remove_inverse_relationships: bool = True,
                            remove_bidirectional_relationships: bool = True) -> None:
    """
    Cleans triplet relationships by replacing inverse relationships, removing bidirectional relationships, and removing duplicates.
    
    Args:
        triplet_filtered_file_path (str): The file path to the filtered triplet dataset.
        triplet_processed_file_path (str): The file path to save the processed triplet dataset.
        relationship_hierarchy_mapping (str, optional) The file path to the hierarchy mapping for relationships. Defaults to None.
        inverse_mapping_path (str, optional) The file path to save the inverse mapping of relationships. Defaults to None.
        reverse_mapping_path (str, optional) The file path to save the reverse mapping of relationships. Defaults to None.
        remove_inverse_relationships (bool): Whether to remove inverse relationships. Defaults to True.
        remove_bidirectional_relationships (bool): Whether to remove bidirectional relationships. Defaults to True.
    
    Returns:
        None
    """
    
    # Load the triplets from the filtered file
    triplets_df = load_triplets(triplet_filtered_file_path)
    
    if remove_inverse_relationships:
        # Process inverse relationships
        rel_inv, remapping = process_inverses_in_triplets(triplet_filtered_file_path,
                                                          relationship_hierarchy_mapping,
                                                          include_self_inv=False)
        
        rel_inv_dict = dict(zip(rel_inv['tail'], rel_inv['head']))
        
        # Substitute multiple inverse options with a single relation
        for i0, row in triplets_df[triplets_df['relation'].isin(remapping.keys())].iterrows():
            triplets_df.loc[i0, 'relation'] = remapping[row['relation']]
        
        # Replace inverse relationships with single relationships
        for i0, row in triplets_df[triplets_df['relation'].isin(rel_inv_dict.keys())].iterrows():
            triplets_df.loc[i0, 'head'] = row['tail']
            triplets_df.loc[i0, 'tail'] = row['head']
            triplets_df.loc[i0, 'relation'] = rel_inv_dict[row['relation']]
        
        save_dict_pandas(rel_inv_dict, inverse_mapping_path)
        save_dict_pandas(remapping, reverse_mapping_path)
    
    if remove_bidirectional_relationships:
        # Remove duplicates and reverse duplicates by creating a unique identifier
        triplets_df['unique_id'] = triplets_df.apply(lambda row: tuple(sorted([row['head'], row['tail']])) + (row['relation'],), axis=1)
        triplets_df = triplets_df.drop_duplicates(subset='unique_id').drop(columns='unique_id')
    else:
        # Remove any exact duplicate triplets
        triplets_df = triplets_df.drop_duplicates()
    
    # Save the processed triplets
    save_triplets(triplets_df, triplet_processed_file_path)


def process_and_merge_missing_triplets(
        missing_triplets_path: str,
        candidates_triplets_path: str,
        triplets_output_path: str,
        nodes_candidates_path: str, 
        relationship_candidates_path: str,
        inverse_mapping_path: str = None,
        reverse_mapping_path: str = None,
        remove_inverse_relationships: bool = True,
        remove_bidirectional_relationships: bool = True) -> None:

    """
    Processes and merges a dataset of missing triplets into an existing candidate set. This function refines the missing triplets by 
    remapping inverse relationships, validating entity and relationship membership, and handling duplicate entries.
    
    Args:
        missing_triplets_path (str): Path to the file containing missing triplets.
        candidates_triplets_path (str): Path to the file containing existing candidate triplets.
        triplets_output_path (str): Path to save the final processed triplets after merging.
        nodes_candidates_path (str): Path to the file containing valid entity (node) candidates.
        relationship_candidates_path (str): Path to the file containing valid relationship candidates.
        inverse_mapping_path (str, optional): Path to the inverse relationship mapping file. Defaults to None.
        reverse_mapping_path (str, optional): Path to the reverse relationship mapping file. Defaults to None.
        remove_inverse_relationships (bool, optional): Whether to remap inverse relationships to a single direction. Defaults to True.
        remove_bidirectional_relationships (bool, optional): Whether to remove bidirectional triplets. Defaults to True.
    
    Returns:
        None: The function outputs the merged and processed triplets into the specified file.
    """
   
    triplets_df = load_triplets(missing_triplets_path)

    if remove_inverse_relationships:
        remapping = load_to_dict(reverse_mapping_path)
        for i0, row in triplets_df[triplets_df['relation'].isin(remapping.keys())].iterrows():
            triplets_df.loc[i0, 'relation'] = remapping[row['relation']]
        
        # Replace inverse relationships with single relationships
        rel_inv_dict = load_to_dict(inverse_mapping_path)
        for i0, row in triplets_df[triplets_df['relation'].isin(rel_inv_dict.keys())].iterrows():
            triplets_df.loc[i0, 'head'] = row['tail']
            triplets_df.loc[i0, 'tail'] = row['head']
            triplets_df.loc[i0, 'relation'] = rel_inv_dict[row['relation']]
    
    # Remove triplets that are not part of the candidate set
    valid_ent = load_to_set(nodes_candidates_path)
    valid_rel = load_to_set(relationship_candidates_path)
    triplets_df = triplets_df[(triplets_df['head'].isin(valid_ent)) &
                (triplets_df['tail'].isin(valid_ent)) &
                (triplets_df['relation'].isin(valid_rel))]
    
    triplets_df = pd.concat([load_triplets(candidates_triplets_path), triplets_df], ignore_index=True)
    
    if remove_bidirectional_relationships:
        # Remove duplicates and reverse duplicates by creating a unique identifier
        triplets_df['unique_id'] = triplets_df.apply(lambda row: tuple(sorted([row['head'], row['tail']])) + (row['relation'],), axis=1)
        triplets_df = triplets_df.drop_duplicates(subset='unique_id').drop(columns='unique_id')
    else:
        # Remove any exact duplicate triplets
        triplets_df = triplets_df.drop_duplicates()
    
    print(f'Final Triplet Size:             {len(triplets_df):>15}')
    # Save the processed triplets
    save_triplets(triplets_df, triplets_output_path)



#------------------------------------------------------------------------------
"""

~~~ Extraction Functions ~~~

These functions extract unique sets of entities or relationships from the triplet dataset.

"""
def _extract_unique_values_from_columns(file_path: str, column_names: list) -> set:
    """
    Extracts a set of unique values from specified columns in the triplet dataset.
    
    Args:
        file_path (str): The file path to the dataset containing the triplets.
        column_names (list): A list of column names from which to extract unique values.
    
    Returns:
        set: A set of unique values found in the specified columns of the triplet dataset.
    """
    
    triplet_df = load_triplets(file_path)
    
    if len(column_names) == 1:
        return set(triplet_df[column_names[0]].tolist())
    else:
        unique_values = set()
        for col in column_names:
            unique_values |= set(triplet_df[col].tolist())
        return unique_values

def extract_triplet_entity_set(file_path: str) -> set:
    """
    Extracts a set of unique entities (heads and tails) from the triplet dataset.
    
    Args:
        file_path (str): The file path to the dataset containing the triplets.
    
    Returns:
        set: A set of unique entities found in the 'head' and 'tail' columns of the triplet dataset.
    """
    
    return _extract_unique_values_from_columns(file_path, ['head', 'tail'])

def extract_triplet_relationship_set(file_path: str) -> set:
    """
    Extracts a set of unique relationships from the triplet dataset.
    
    Args:
        file_path (str): The file path to the dataset containing the triplets.
    
    Returns:
        set: A set of unique relationships found in the 'relation' column of the triplet dataset.
    """
    
    return _extract_unique_values_from_columns(file_path, ['relation'])

def extract_triplet_sets(triplet_processed_file_path: str, 
                         nodes_candidates_path: str, 
                         relationship_candidates_path: str, 
                         nodes_missing_path: str = None, 
                         triplet_file_path: str = None) -> None:
    """
    Extracts and saves key sets from the processed triplet dataset, including the entity (node) set, 
    relationship set, and optionally the missing entity set if an original triplet file is provided.

    Args:
        triplet_processed_file_path (str): The file path to the processed triplet dataset.
        nodes_candidates_path (str): The path to save the extracted entity candidates.
        relationship_candidates_path (str): The path to save the extracted relationship candidates.
        nodes_missing_path (str, optional): The path to save the missing entities. Defaults to None.
        triplet_file_path (str, optional): The file path to the original triplet dataset. Defaults to None.

    Returns:
        None
    """
    # Load the processed triplets
    triplets_df = load_triplets(triplet_processed_file_path)
    
    # Extract the set of entities (nodes)
    entity_set = extract_triplet_entity_set(triplet_processed_file_path)
    
    # Extract the set of relationships
    relationship_set = extract_triplet_relationship_set(triplet_processed_file_path)
    
    # Save the extracted entity and relationship sets
    save_set_pandas(entity_set, nodes_candidates_path)
    save_set_pandas(relationship_set, relationship_candidates_path)
    
    # Optionally calculate and save the missing entity set if original triplet file is provided
    if triplet_file_path and nodes_missing_path:
        missing_entity_set = find_missing_entities(triplet_file_path, triplet_processed_file_path)
        save_set_pandas(missing_entity_set, nodes_missing_path)
    
    # Print the statistics
    print(f'Number of Triplets:             {len(triplets_df):>15}')
    print(f'Number of Entities (Nodes):     {len(entity_set):>15}')
    print(f'Number of Relationship Types:   {len(relationship_set):>15}')
    
    if triplet_file_path and nodes_missing_path:
        print(f'Number of Missing Entities:     {len(missing_entity_set):>15}')


#------------------------------------------------------------------------------
"""

~~~ Split Functions ~~~

These functions handle splitting the dataset into training, testing, and validation sets.

"""
def split_triplets(file_path: str, train_path: str, test_path: str, valid_path: str, split_rate:float = 0.8, veborse: bool = True) -> None:
    """
    Splits triplets from a file into training, testing, and validation sets.
    
    Args:
        file_path (str): The path to the file containing the triplets.
        train_path (str): The path to save the training set.
        test_path (str): The path to save the testing set.
        valid_path (str): The path to save the validation set.
        split_rate (float): The proportion of data to use for the training set. Must be between 0.0 and 1.0.
        verbose (bool): Whether to print statistics. Defaults to True.
    Returns:
        None
    """
    
    assert ((split_rate > 0.0) and (split_rate < 1.0)), 'Error! The split rate must be between 0.0 and 1.0!'
    
    test_rate = (1 - split_rate)/2
    
    # Load the triplets into a DataFrame
    triplets_df = load_triplets(file_path)
    
    # Shuffle the DataFrame
    triplets_df = triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the split indices
    train_end = int(split_rate * len(triplets_df))
    test_end = train_end + int(test_rate * len(triplets_df))
    
    # Split the DataFrame
    train_df = triplets_df.iloc[:train_end]
    test_df = triplets_df.iloc[train_end:test_end]
    valid_df = triplets_df.iloc[test_end:]
    
    print(f'Train Size:                     {len(train_df):>15.0f}')
    print(f'Test Size:                      {len(test_df):>15.0f}')
    print(f'Valid Size:                     {len(valid_df):>15.0f}')
    
    # Save the splits to their respective files
    save_triplets(train_df, train_path)
    save_triplets(test_df, test_path)
    save_triplets(valid_df, valid_path)