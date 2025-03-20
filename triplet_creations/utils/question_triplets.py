# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:17:21 2024

@author: Eduin Hernandez
"""
import re

import pandas as pd

from typing import List

def prepare_prompt(question: str, entities: List[str], relations: List[str], descriptions: List[str]) -> str:
    """
    Creates a prompt to extract relationships from a given question.
    
    Args:
        question (str): The jeopardy question.
        entities (list): List of entities involved.
        relations (list): List of relationships.
        descriptions (list): List of relationship descriptions.
    
    Returns:
        str: The formatted prompt.
    """
    
    relations_with_descriptions = '\n'.join([f"- {rel}: {desc}" for rel, desc in zip(relations, descriptions)])
    prompt = f"""
    ***Jeopardy Evaluation***
    Below, I will provide:
    1. A set of questions.
    2. A list of possible names.
    3. A list of valid relationships.
    4. A list of description for each relationship

    Your task is to extract all triplets from each question, following these guidelines:
    1. **Triplet Structure**: Format each triplet as [entity1, relationship, entity2], where:
        - `entity1` and `entity2` are derived from the question or the provided list.
        - The `relationship` must come from the provided list of valid relationships and be explicitly or implicitly present in the question.
        - List all possible triplets as long as `entity1` and `entity2` are consistent, and the `relationships` differ.

    2. **Handling Irrelevant Entities and Relationships**:
        - Exclude any entity or relationship that is not contextually relevant to the question.
        - Always include `(Unknown Key)` in at least one triplet as a placeholder for an unknown entity.
        
    3. **(Unknown Key) Placeholder**:
        - Use `(Unknown Key)` when an entity is unknown, but ensure it forms a logical connection.
        - `(Unknown Key)` cannot be both `entity1` and `entity2` in the same triplet.
        
    4. **Extracting Logical Relationships**:
        - Ensure extracted relationships are logical, contextually relevant, and drawn from the provided list.
        - Prioritize relationships in the provided list based on their relevance.
        - Use the descriptions to help determine the most appropriate relationship for the context.

    **Response Format**:
    Return the triplets in a comma-separated format without any additional text or formatting, as shown below:
    `[entity1, relationship, entity2], [entity1, relationship, entity2], ... `
    ---
    Question: {question}
    Entities: {entities}
    Relationships: {relations}
    Description: {relations_with_descriptions}
    """

    return prompt

def extract_triplets(response: str):
    """
    Extracts triplets from the GPT model's response.
    
    Args:
        response (str): Response from GPT containing triplets.
    
    Returns:
        list: A list of extracted triplets.
    """
    triplets = re.findall(r'\[(.*?)\]', response)
    result = [triplet.split(', ') for triplet in triplets if len(triplet.split(', ')) == 3]
    result = [[item[1:-1] if item.startswith(('`', '"')) and item.endswith(('`', '"')) else item for item in triplet] for triplet in result]
    return result


def titles2ids(df: pd.DataFrame, node_df: pd.DataFrame, rel_df: pd.DataFrame, new_row: pd.DataFrame, q_ids: List[str], p_ids: List[str]) -> pd.DataFrame:
    """
    Maps entity and relationship titles in a DataFrame to their corresponding IDs, and filters the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing triplets with columns ['head', 'relation', 'tail'].
        node_df (pd.DataFrame): DataFrame containing entities with columns ['QID', 'Title'].
        rel_df (pd.DataFrame): DataFrame containing relationships with columns ['Property', 'Title'].
        new_row (pd.DataFrame): DataFrame containing 'Unknown' entity rows to be added to node_df.
        q_ids (List[str]): List of entity IDs for valid nodes.
        p_ids (List[str]): List of property IDs for valid relationships.

    Returns:
        pd.DataFrame: Filtered DataFrame where titles have been replaced by corresponding IDs and duplicates have been removed.
    """
    valid_nodes = node_df[node_df['QID'].isin(q_ids)].copy()
    valid_nodes = pd.concat([valid_nodes, new_row], ignore_index=True).fillna('')
    valid_nodes_map = valid_nodes.drop_duplicates(subset='Title').set_index('Title')['QID']
    
    valid_rels = rel_df[rel_df['Property'].isin(p_ids)].copy()
    valid_rel_map = valid_rels.set_index('Title')['Property']
    
    # Replace head and tail with QID values
    df['head'] = df['head'].map(valid_nodes_map).fillna(df['head'])
    df['tail'] = df['tail'].map(valid_nodes_map).fillna(df['tail'])
    df['relation'] = df['relation'].map(valid_rel_map).fillna(df['relation'])
    
    df = filter_triplet(df)
    
    # Remove duplicate triplets
    return df.drop_duplicates()

def filter_triplet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame of triplets by ensuring valid QID node values and relations.
    
    Args:
        df (pd.DataFrame): DataFrame containing triplets with columns 'head', 'relation', and 'tail'.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only valid triplets based on the conditions.
    """

    # Drop any head or tail that do not start with 'Q' or 'Unknown'
    df = df[df['head'].str.startswith(('Q', 'Unknown')) & df['tail'].str.startswith(('Q', 'Unknown'))]

    # Drop rows where both head and tail are 'Unknown'
    df = df[~((df['head'] == 'Unknown') & (df['tail'] == 'Unknown'))]
    
    return df[df['relation'].str.startswith(('P'))]