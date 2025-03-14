# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:02:11 2024

@author: Eduin Hernandez

Summary:
The `basic` module provides a collection of utility functions for loading, saving, sorting, and processing data.
It includes tools for working with JSON, pandas DataFrames, sets, dictionaries, and handling triplet data 
(often used in knowledge graph representations). Additionally, the package provides argument parsing tools, 
string conversion utilities, and functions for extracting and manipulating data from columns containing string literals.

Core functionalities:
- **Loading & Saving Functions**: Load and save data in various formats (JSON, CSV, triplets) for efficient data management.
- **Sorting Functions**: Handle sorting of QID-based data and dictionaries.
- **Argument Parsing**: Convert string inputs to booleans for command-line argument parsing.
- **Pandas Utilities**: Extract lists of string literals from pandas Series and optionally flatten them into a single list.

This package is intended to be used as a foundational toolkit for loading, processing, and managing data in larger projects.
"""
import argparse
import ast
import yaml

import json
import pandas as pd

from typing import List, Dict, Union

#------------------------------------------------------------------------------
'Loading & Saving Functions'

def load_json(file_path: str) -> Dict[str, any]:
    """
    Loads a JSON file from the specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, any]: The loaded JSON data as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_pandas(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(file_path).fillna('')

def load_to_set(file_path: str) -> set:
    """
    Loads a text file and converts it to a set of values.

    Args:
        file_path (str): The path to the text file.

    Returns:
        set: A set of unique values from the file.
    """
    df = pd.read_csv(file_path, header=None, names=['qid'])
    return set(df['qid'])

def load_to_dict(file_path: str) -> dict:
    """
    Loads a tab-separated text file and converts it into a dictionary.

    Args:
        file_path (str): The path to the tab-separated text file.

    Returns:
        dict: A dictionary with keys and values from the file.
    """
    df = pd.read_csv(file_path, sep='\t')
    return dict(zip(df['Key'], df['Value']))

def load_triplets(file_path: Union[str, List[str]]) -> pd.DataFrame():
    """
    Loads a triplet dataset from one or more file paths into a pandas DataFrame.

    Args:
        file_path (str or list): The path or list of paths to the triplet files.

    Returns:
        pd.DataFrame: A DataFrame containing the triplet data (head, relation, tail).
    """
    if type(file_path) == str:
        # Load the triplets into a DataFrame
        return pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    elif type(file_path) == list:
        # Load and merge the DataFrames from the list of file paths
        df_list = [pd.read_csv(fp, sep='\t', header=None, names=['head', 'relation', 'tail']) for fp in file_path]
        return pd.concat(df_list, ignore_index=True)
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'

def load_embeddings(file_path: str) -> pd.DataFrame:
    """
    Loads embeddings from a CSV file, where the first column is 'Property' and the remaining columns 
    represent embedding values.
    
    The function skips the first row, renames the first column to 'Property', and merges the remaining columns 
    into a list to create a single 'Embedding' column.
    
    Args:
        file_path (str): Path to the CSV file containing embeddings.
    
    Returns:
        pd.DataFrame: DataFrame with two columns:
            - 'Property': The identifier or label for each embedding.
            - 'Embedding': A list of embedding values as a single column.
    """
    # Load the CSV file without a header and skip the first row
    df = pd.read_csv(file_path, header=None, skiprows=1)

    df.rename(columns={0: 'Property'}, inplace=True)

    # Merge the remaining columns into a list for each row
    df['Embedding'] = df.iloc[:, 1:].apply(lambda row: row.tolist(), axis=1)
    
    # Drop the original separate embedding columns if they’re no longer needed
    df = df[['Property', 'Embedding']]
    return df

def save_set_pandas(pd_set: set, file_path: str) -> None:
    """
    Saves a set to a text file using pandas.

    Args:
        pd_set (set): The set containing the information to save.
        file_path (str): The path to store the text file.
    """
    
    df = pd.DataFrame(pd_set, columns=['Items'])
    
    df = sort_by_qid(df, column_name='Items')
    
    df.to_csv(file_path, index=False, header=False)
    
def save_dict_pandas(pd_dict: set, file_path: str) -> None:
    """
    Saves a dictionary to a tab-separated text file using pandas.

    Args:
        pd_dict (dict): The dictionary to save.
        file_path (str): The path to store the text file.
    """
    df = pd.DataFrame(list(pd_dict.items()), columns=['Key', 'Value'])

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, sep='\t', index=False, header=True)
    
def save_triplets(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas DataFrame containing triplets (head, relation, tail) to a tab-separated text file.

    Args:
        df (pd.DataFrame): The DataFrame containing the triplet data to save.
        file_path (str): The path to save the file as a tab-separated text file.
    """
    df.to_csv(file_path, sep='\t', header=False, index=False)

#------------------------------------------------------------------------------
'Sorting Functions'

def revert_dict(dt):
    """
    Reverses the keys and values in a dictionary.
    
    Args:
        dt (dict): The dictionary to be reversed.
    
    Returns:
        dict: The dictionary with keys and values swapped.
    """
    return {value: key for key, value in dt.items()}

def sort_by_qid(df: pd.DataFrame, column_name: str = 'qid') -> pd.DataFrame:
    """
    Sorts a DataFrame based on the numeric part of a specified column containing QIDs.

    Args:
        df (pd.DataFrame): The DataFrame to be sorted.
        column_name (str): The name of the column to sort by.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    
    df['numeric_part'] = df[column_name].str.extract('(\d+)').astype(int)
    return df.sort_values(by='numeric_part').drop(columns='numeric_part')

def sort_qid_list(qid_list: list) -> list:
    """
    Sorts a list of QIDs by their numeric parts.

    Args:
        qid_list (list): The list of QIDs to be sorted.

    Returns:
        list: The sorted list of QIDs.
    """
    df = pd.DataFrame(qid_list, columns=['qid'])
    sorted_df = sort_by_qid(df)
    return sorted_df['qid'].tolist()

def sort_json_by_keys(json_data: Dict[str, any]) -> Dict[str, any]:
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
'Argparser'

def str2bool(string):
    """
    Converts a string input to a boolean value.
    
    Args:
        string (str): The string to convert ('yes', 'true', 'no', 'false', etc.).
    
    Returns:
        bool: The corresponding boolean value.
    
    Raises:
        argparse.ArgumentTypeError: If the input cannot be converted to a boolean.
    """
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif string.lower() in ('none'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#------------------------------------------------------------------------------
'Pandas'

def random_dataframes(df: pd.DataFrame, n: int, random_state: int = None) -> pd.DataFrame:
    """
    Randomly selects a specified number of rows from the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which rows will be randomly selected.
        n (int): The number of rows to randomly select.
        random_state (int): The seed value for random number generation to ensure reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing `n` randomly selected rows from the original DataFrame.
    """
    return df.sample(n=n, random_state=random_state)

def extract_literals(column: Union[str, pd.Series], flatten: bool = False) -> Union[pd.Series, List[str]]:
    """
    Extracts the list of string literals from each entry in the provided column (Pandas Series or string)
    using ast.literal_eval. Optionally flattens the extracted lists into a single list if 'flatten' is set to True.

    Args:
        column (Union[str, pd.Series]): The column containing string representations of lists. Can be a
                                        Pandas Series or a string representation of a list.
        flatten (bool): If True, flattens the lists into a single list. Default is False.

    Returns:
        Union[pd.Series, List[str]]: A Pandas Series of lists if flatten is False, otherwise a single flattened list of strings.
    """
    # Convert the input to a Pandas Series if it's a string
    if isinstance(column, str):
        column = pd.Series([column])

    # Convert string representations of lists into actual Python lists
    column = column.apply(ast.literal_eval)

    # Flatten the lists if the flatten argument is True
    if flatten: column = [item for sublist in column for item in sublist]

    return column

def overload_parse_defaults_with_yaml(yaml_location:str, args: argparse.Namespace) -> argparse.Namespace:
    with open(yaml_location, "r") as f:
        yaml_args = yaml.load(f, Loader=yaml.FullLoader)
        overloaded_args = recurse_until_leaf(yaml_args)
        for k, v in overloaded_args.items():
            if k in args.__dict__:
                setattr(args, k, v)
            else:
                raise ValueError(f"Key {k} not found in args")
    return args

def recurse_until_leaf(d: dict, parent_key: str = "") -> dict:
    return_dict = {}
    for k, v in d.items():
        next_key = f"{parent_key}_{k}" if parent_key != "" else k
        if isinstance(v, dict):
            deep_dict = recurse_until_leaf(v, parent_key=next_key)
            return_dict.update(deep_dict)
        else:
            return_dict[next_key] = v
    return return_dict
