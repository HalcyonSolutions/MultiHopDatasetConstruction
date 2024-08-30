# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:06:49 2024

@author: Eduin Hernandez
"""
import ast

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
'Statistics Functions'
def calculate_statistics(triplet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency statistics for nodes and relationships in the triplets.

    Args:
        triplet_df (pd.DataFrame): DataFrame containing triplet data with columns ['head', 'relation', 'tail'].

    Returns:
        pd.DataFrame: DataFrame with statistics for nodes and relationships.
    """

    # Frequency of nodes as head
    head_freq = triplet_df['head'].value_counts().reset_index()
    head_freq.columns = ['node', 'head_count']

    # Frequency of nodes as tail
    tail_freq = triplet_df['tail'].value_counts().reset_index()
    tail_freq.columns = ['node', 'tail_count']

    # Combine head and tail frequencies
    combined_freq = pd.merge(head_freq, tail_freq, on='node', how='outer').fillna(0)
    combined_freq['total_count'] = combined_freq['head_count'] + combined_freq['tail_count']

    # Frequency of relationships
    relation_freq = triplet_df['relation'].value_counts().reset_index()
    relation_freq.columns = ['relation', 'relation_count']

    return combined_freq, relation_freq

def count_relationships_per_node(triplet_df: pd.DataFrame, nodes: pd.DataFrame, rels: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table that counts the number of times each relationship (Property) appears for each node (RDF),
    considering the node as both head and tail.

    Args:
        triplet_df (pd.DataFrame): DataFrame containing triplet data with columns ['head', 'relation', 'tail'].
        nodes (pd.DataFrame): DataFrame containing node information with columns ['RDF', 'Title'].
        rels (pd.DataFrame): DataFrame containing relationship details with columns ['Property', 'Title'].

    Returns:
        pd.DataFrame: Pivot table where rows represent nodes (RDF) and columns represent relationships (Property),
                      with values indicating the count of each relationship per node.
    """

    # Consider both head and tail roles
    head_df = triplet_df.merge(nodes[['RDF']], left_on='head', right_on='RDF', how='left')\
                        .merge(rels[['Property']], left_on='relation', right_on='Property', how='left')
    
    tail_df = triplet_df.merge(nodes[['RDF']], left_on='tail', right_on='RDF', how='left')\
                        .merge(rels[['Property']], left_on='relation', right_on='Property', how='left')

    # Combine the head and tail dataframes
    combined_df = pd.concat([head_df[['RDF', 'Property']], tail_df[['RDF', 'Property']]])

    # Group by RDF and Property and count occurrences
    counts_df = combined_df.groupby(['RDF', 'Property']).size().unstack(fill_value=0)

    return counts_df
#------------------------------------------------------------------------------
'Mapping Functions'
def map_nodes_to_categories(relationship_count_per_node: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the relationship_count_per_node DataFrame by mapping the nodes to their respective categories.
    
    Args:
        relationship_count_per_node (pd.DataFrame): DataFrame where rows represent nodes (RDF) and columns represent 
                                                    relationships (Property), with values indicating counts.
        nodes_df (pd.DataFrame): DataFrame containing node information with columns ['RDF', 'Title', 'Category'].
    
    Returns:
        pd.DataFrame: Modified DataFrame with rows mapped to the respective categories, combined where necessary.
    """

    # Initialize a list to store the modified rows
    modified_rows = []

    # Iterate over each row in the relationship_count_per_node DataFrame
    for rdf_value, row in relationship_count_per_node.iterrows():
        # Find the corresponding category in the nodes DataFrame
        category_list = ast.literal_eval(nodes_df.loc[nodes_df['RDF'] == rdf_value, 'Category'].values[0])
        # Apply the rules based on the category list
        if not category_list:  # 1-a: Empty category list
            modified_rows.append(row.rename(rdf_value))
        elif len(category_list) == 1:  # 1-b: Single category entry
            modified_rows.append(row.rename(category_list[0]))
        else:  # 1-c: Multiple category entries
            for category in category_list:
                modified_rows.append(row.rename(category))

    # Concatenate all the modified rows into a single DataFrame
    modified_df = pd.concat(modified_rows, axis=1).T
    
    # Group by 'RDF' (the new index) and sum the counts to combine duplicates
    modified_df = modified_df.groupby(modified_df.index).sum()

    return modified_df

#------------------------------------------------------------------------------
'Plotting Functions'
def plot_relationship_statistics(relation_stats: pd.DataFrame, rels: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> None:
    """
    Plot the frequency statistics for relationships with titles for relationships.

    Args:
        relation_stats (pd.DataFrame): DataFrame containing relation frequency statistics.
        rels (pd.DataFrame): DataFrame containing relationship titles mapped by property.
        start_idx (int, optional): The starting index for the relationships to display. Defaults to 0.
        end_idx (int, optional): The ending index for the relationships to display. Defaults to None (shows all).
    """

    # Map relationship properties to their titles
    relation_stats = relation_stats.merge(rels[['Property', 'Title']], left_on='relation', right_on='Property', how='left')
    relation_stats.drop(columns=['Property'], inplace=True)

    # Sort by total_count in descending order
    relation_stats = relation_stats.sort_values(by='relation_count', ascending=False)

    # Plot for relationships with titles and flipped axes
    plt.figure(figsize=(10, 12))
    plt.barh(relation_stats['Title'][start_idx:end_idx], relation_stats['relation_count'][start_idx:end_idx], color='lightgreen')
    plt.ylabel('Relationship Title')
    plt.xlabel('Frequency')
    plt.title('Relationship Frequency')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_node_statistics(node_stats: pd.DataFrame, nodes: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> None:
    """
    Plot the frequency statistics for nodes with titles for nodes.

    Args:
        node_stats (pd.DataFrame): DataFrame containing node frequency statistics.
        nodes (pd.DataFrame): DataFrame containing node titles mapped by RDF.
        start_idx (int, optional): The starting index for the relationships to display. Defaults to 0.
        end_idx (int, optional): The ending index for the relationships to display. Defaults to None (shows all).
    """

    # Map node RDF to their titles
    node_stats = node_stats.merge(nodes[['RDF', 'Title']], left_on='node', right_on='RDF', how='left')
    node_stats.drop(columns=['RDF'], inplace=True)

    # Sort by total_count in descending order
    node_stats = node_stats.sort_values(by='total_count', ascending=False)

    # Plot for nodes with titles
    plt.figure(figsize=(10, 12))
    plt.barh(node_stats['Title'][start_idx:end_idx], node_stats['total_count'][start_idx:end_idx], color='skyblue')
    plt.ylabel('Node Title')
    plt.xlabel('Frequency')
    plt.title('Node Frequency')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
def plot_distinct_edges(mat: pd.DataFrame, nodes: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> None:
    """
    Plots the number of distinct relationship types (edges) each node is involved in, considering all relationships.
    
    This function takes a matrix of relationships and nodes, calculates the number of distinct relationship types 
    each node is connected to, and then plots these counts. The plot is ordered by the number of distinct edges, 
    and you can specify a range of nodes to display.
    
    Args:
        mat (pd.DataFrame): DataFrame where rows represent nodes (RDF) and columns represent relationship properties.
                            Each entry represents the count of a particular relationship for a node.
        nodes (pd.DataFrame): DataFrame containing node information with columns ['RDF', 'Title'].
        start_idx (int, optional): Starting index for nodes to display in the plot. Defaults to None.
        end_idx (int, optional): Ending index for nodes to display in the plot. Defaults to None.
    """
    unique_relationships = (mat > 0).astype(int)
    
    # Sum along the rows (columns) to get the number of distinct nodes per relationship
    distinct_rel_counts = unique_relationships.sum(axis=1)
    
    # Convert the Series to a DataFrame and reset the index to turn it into a column
    distinct_rel_counts_df = distinct_rel_counts.reset_index()
    distinct_rel_counts_df.columns = ['RDF', 'count']
    node_df = distinct_rel_counts_df.merge(nodes[['RDF', 'Title']], on='RDF', how='left')
    
    # Sort by total_count in descending order
    node_df = node_df.sort_values(by='count', ascending=False)
    
    # Plot for nodes with titles
    plt.figure(figsize=(10, 12))
    plt.barh(node_df['Title'][start_idx:end_idx], node_df['count'][start_idx:end_idx], color='skyblue')
    plt.ylabel('Node Title')
    plt.xlabel('Number of Edge Types')
    plt.title('Relationship Types per Node')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_distinct_nodes(mat: pd.DataFrame, rels: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> None:
    """
    Plots the number of distinct nodes each relationship (property) is connected to, considering all nodes.
    
    This function takes a matrix of relationships and nodes, calculates the number of distinct nodes each 
    relationship is involved with, and then plots these counts. The plot is ordered by the number of distinct 
    nodes, and you can specify a range of relationships to display.
    
    Args:
        mat (pd.DataFrame): DataFrame where rows represent nodes (RDF) and columns represent relationship properties.
                            Each entry represents the count of a particular relationship for a node.
        rels (pd.DataFrame): DataFrame containing relationship information with columns ['Property', 'Title'].
        start_idx (int, optional): Starting index for relationships to display in the plot. Defaults to None.
        end_idx (int, optional): Ending index for relationships to display in the plot. Defaults to None.
    """
    unique_relationships = (mat > 0).astype(int)
    
    # Sum along the rows (columns) to get the number of distinct nodes per relationship
    distinct_node_counts = unique_relationships.sum(axis=0)
    
    distinct_node_count_df = distinct_node_counts.reset_index()
    distinct_node_count_df.columns = ['Property', 'count']
    rels_df = distinct_node_count_df.merge(rels[['Property', 'Title']], on='Property', how='left')
    
    # Sort by total_count in descending order
    rels_df = rels_df.sort_values(by='count', ascending=False)
    
    # Plot for nodes with titles
    plt.figure(figsize=(10, 12))
    plt.barh(rels_df['Title'][start_idx:end_idx], rels_df['count'][start_idx:end_idx], color='lightgreen')
    plt.ylabel('Relationship Title')
    plt.xlabel('Number of Node Types')
    plt.title('Node Types per Relationship')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
def plot_category_node_statistics(category: pd.DataFrame, nodes: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> None:
    """
    Plot the frequency of categories across nodes, showing how many times each category is associated with nodes.

    This function calculates the total occurrences of categories across all nodes, sorts them in descending order, 
    and plots a horizontal bar chart. The chart visualizes the frequency of categories across nodes, with an option 
    to specify a range of nodes to display.

    Args:
        category (pd.DataFrame): DataFrame where rows represent nodes (RDF) and columns represent categories, with values 
                                 indicating the presence of a category for each node.
        nodes (pd.DataFrame): DataFrame containing node information with columns ['RDF', 'Title'].
        start_idx (int, optional): Starting index for nodes to display in the plot. Defaults to None.
        end_idx (int, optional): Ending index for nodes to display in the plot. Defaults to None.
    """    
    # Sum along the rows (columns) to get the number of distinct nodes per relationship
    category = category.sum(axis=1)
    
    # Convert the Series to a DataFrame and reset the index to turn it into a column
    category = category.reset_index()
    category.columns = ['RDF', 'count']
    node_df = category.merge(nodes[['RDF', 'Title']], on='RDF', how='left')
    
    # Sort by total_count in descending order
    node_df = node_df.sort_values(by='count', ascending=False)
    
    # Plot for nodes with titles
    plt.figure(figsize=(10, 12))
    plt.barh(node_df['Title'][start_idx:end_idx], node_df['count'][start_idx:end_idx], color='skyblue')
    plt.ylabel('Category Title')
    plt.xlabel('Frequency')
    plt.title('Category Frequency')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
def plot_category_relationship_statistics(category: pd.DataFrame, rels: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> None:
    """
    Plot the frequency of relationships across categories, showing how many times each relationship is associated with categories.

    This function calculates the total occurrences of relationships across all categories, sorts them in descending order, 
    and plots a horizontal bar chart. The chart visualizes the frequency of relationships per category, with an option 
    to specify a range of relationships to display.

    Args:
        category (pd.DataFrame): DataFrame where rows represent nodes (RDF) and columns represent categories, with values 
                                 indicating the presence of a category for each node.
        rels (pd.DataFrame): DataFrame containing relationship information with columns ['Property', 'Title'].
        start_idx (int, optional): Starting index for relationships to display in the plot. Defaults to None.
        end_idx (int, optional): Ending index for relationships to display in the plot. Defaults to None.
    """    
    # Sum along the rows (columns) to get the number of distinct nodes per relationship
    category = category.sum(axis=0)
    
    # Convert the Series to a DataFrame and reset the index to turn it into a column
    category = category.reset_index()
    category.columns = ['Property', 'count']
    node_df = category.merge(rels[['Property', 'Title']], on='Property', how='left')
    
    # Sort by total_count in descending order
    rel_df = node_df.sort_values(by='count', ascending=False)
    
    # Plot for nodes with titles
    plt.figure(figsize=(10, 12))
    plt.barh(rel_df['Title'][start_idx:end_idx], rel_df['count'][start_idx:end_idx], color='lightgreen')
    plt.ylabel('Relationship Title')
    plt.xlabel('Frequency')
    plt.title('Relationship Frequency per Category')
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_zipfs_law(frequencies: list[float], title: str) -> None:
    """
    Plot the frequency distribution of relationships to demonstrate Zipf's Law, and overlay a theoretical Zipf's Law line.

    This function receives a list of frequencies, sorts them in descending order, and plots a line chart on a logarithmic scale. 
    The plot illustrates Zipf's Law by showing the relationship between rank and frequency, and overlays a line that represents
    the expected Zipf's Law distribution for comparison.

    Args:
        frequencies (list[float]): List of frequencies, typically corresponding to relationships.
        title (str): Title for the plot, indicating the context or dataset being analyzed.
    """

    # Ensure that the frequencies are sorted in descending order
    frequencies = sorted(frequencies, reverse=True)

    # Calculate the rank
    ranks = np.arange(1, len(frequencies) + 1)

    # Plotting the frequencies
    plt.figure(figsize=(10, 12))
    plt.plot(ranks, frequencies, marker='o', linestyle='-', color='lightgreen', label='Observed Frequencies')
    
    # Plotting the Zipf's Law line (a straight line on a log-log scale)
    zipfs_law_line = frequencies[0] / ranks  # Ideal Zipf's Law line
    plt.plot(ranks, zipfs_law_line, color='red', linestyle='--', label="Zipf's Law")

    # Set log-log scale
    plt.xscale('log')
    plt.yscale('log')

    # Labels and title
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f"Zipf's Law for {title}")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()