# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:02:11 2024

@author: Eduin Hernandez

Summary: Basic Tools
"""

import json
import pandas as pd

from typing import Dict

#------------------------------------------------------------------------------

def load_json(file_path: str) -> Dict[str, any]:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_pandas(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

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

#------------------------------------------------------------------------------
'Sorting Functions'
def revert_dict(dt):
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