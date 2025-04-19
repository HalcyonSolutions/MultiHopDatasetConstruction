# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:18:44 2024

@author: Eduin Hernandez

Summary:
This package provides tools to scrape, process, and retrieve data from Wikidata, focusing on both entities and relationships. 
It handles concurrent data fetching using threading to speed up the retrieval process, making it efficient for working with 
large datasets of entities and relationships. Additionally, it includes utility functions for retrying failed fetches and logging failures.

Core functionalities:
- **Entity Data Processing**: Fetch and process detailed information about entities from Wikidata.
- **Entity Triplet Processing**: Retrieve triplet relationships between entities.
- **Relationship Data Processing**: Extract relationships and their hierarchical structures.
- **Concurrency**: Use of multithreading to speed up data retrieval processes.
- **Retry and Timeout Handling**: Built-in retry mechanisms to handle network or API errors during data fetching.
"""

import math
import pandas as pd

import time
import requests
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from wikidata.client import Client

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Union

from utils.basic import load_to_set, sort_by_qid, sort_qid_list

#------------------------------------------------------------------------------
'Webscrapping for Entity Info'

def update_entity_data(entity_df: pd.DataFrame, missing_entities: list, max_workers: int = 10,
                       max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_ent_log.txt') -> pd.DataFrame:
    """
    Fetches additional entity details concurrently for a list of entities from Wikidata and appends them to the provided DataFrame.

    Args:
        entity_df (pd.DataFrame): DataFrame containing existing entity data.
        missing_entities (list): List of entity identifiers that need to be fetched.
        max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed fetches. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch. Defaults to 2.
        verbose (bool, optional): Print error details. Defaults to False.
        failed_log_path (str, optional): Path to log failed entity retrievals. Defaults to './data/failed_ent_log.txt'.

    Returns:
        pd.DataFrame: The combined DataFrame with the newly fetched entity data appended to the existing data.
    """
    
    entity_list = list(missing_entities)
    entity_list_size = len(entity_list)

    results_template = {
        'RDF': '',
        'Title': '',
        'Description': '',
        'Alias': '',
        'MDI': '',
        'URL': '',
        'Forwarding': '',
    }
    
    data = []
    failed_entities = []

    client = Client()

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_entity_details, entity_list[i], results_template, client,
                                   max_retries=max_retries, timeout=timeout, verbose=verbose): i for i in range(entity_list_size)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching Entity Data'):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                data.append(result)
            except HTTPError as http_err:
                failed_entities.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                failed_entities.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                failed_entities.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"Error: {e}")
    
    # Save failed entities to a log file
    if failed_entities:
        with open(failed_log_path, 'w') as log_file:
            for entity in failed_entities:
                log_file.write(f"{entity}\n")

    #--------------------------------------------------------------------------
    df = pd.DataFrame(data)
    
    df = df.dropna(subset=['RDF'])
    df = df[df['RDF'].str.strip() != '']  # Then, remove empty strings

    missing = set(entity_list) - set(df['RDF'].tolist())
    
    # Create a DataFrame from the set with empty values for other columns
    new_rows = pd.DataFrame([[rdf] + ['']*(len(results_template)-1) for rdf in missing], columns=df.columns)
    
    # Append new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    combined_df = pd.concat([entity_df, df], ignore_index=True)
    combined_df.drop_duplicates(subset='RDF', inplace=True)
    
    # Sort the DataFrame by the "RDF" column
    combined_df = sort_by_qid(combined_df, column_name = 'RDF')

    return combined_df

def process_entity_data(file_path: Union[str, List[str]], output_file_path: str, nrows: int = None, max_workers: int = 10,
                        max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_ent_log.txt') -> None:
    """
    Processes entity data by fetching details from Wikidata for each entity and saving the results to a CSV file.

    Args:
        file_path (str or list): Path to the file or a list of files containing entity IDs.
        output_file_path (str): Path to save the processed CSV file.
        nrows (int, optional): Number of rows to process (None for all). Defaults to None.
        max_workers (int, optional): Maximum number of threads for parallel processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed requests. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch request. Defaults to 2.
        verbose (bool, optional): Print additional error information. Defaults to False.
        failed_log_path (str, optional): Path to save a log of failed entity retrievals. Defaults to './data/failed_ent_log.txt'.

    Returns:
        None: The function saves the processed data to a CSV file.
    """
    
    if type(file_path) == str:
        entity_list = list(load_to_set(file_path))[:nrows]
    elif type(file_path) == list:
        entity_list = set()
        for file in file_path:
            entity_list.update(load_to_set(file_path))
        entity_list = list(entity_list)[:nrows]
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'
        
    entity_list_size = len(entity_list)

    results_template = {
        'RDF': '',
        'Title': '',
        'Description': '',
        'Alias': '',
        'MDI': '',
        'URL': '',
        'Forwarding': '',
    }

    data = []
    failed_ents = []

    client = Client()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_entity_details, entity_list[i0], results_template, client,
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i0 for i0 in range(0, entity_list_size)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Entities Data"):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                data.append(result)
            except HTTPError as http_err:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"Error: {e}")

    # Save failed entities to a log file
    if failed_ents:
        with open(failed_log_path, 'w') as log_file:
            for ent in failed_ents:
                log_file.write(f"{ent}\n")
                
    df = pd.DataFrame(data)
    
    missing = set(entity_list) - set(df['RDF'].tolist())
    
    # Create a DataFrame from the set with empty values for other columns
    new_rows = pd.DataFrame([[rdf] + ['']*(len(results_template)-1) for rdf in missing], columns=df.columns)
    
    # Append new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    
    df.drop_duplicates(subset='RDF', inplace=True)
    
    # Sort the DataFrame by the "RDF" column
    df = sort_by_qid(df, column_name = 'RDF')
    
    # Save the updated and sorted DataFrame
    df.to_csv(output_file_path, index=False)
    print("\nData processed and saved to", output_file_path)

def process_entity_triplets(file_path: Union[str, List[str]], output_file_path: str, nrows: int = None, max_workers: int = 10,
                            max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_ent_log.txt') -> pd.DataFrame:
    """
    Scrapes and processes triplet relationships for a set of entities from Wikidata and saves the data to a TXT file.

    Args:
        file_path (str or list): Path to the input file or list of files with entity IDs.
        output_file_path (str): Path to save the processed triplets.
        nrows (int, optional): Number of rows to process (None for all). Defaults to None.
        max_workers (int, optional): Maximum number of threads for parallel processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed fetch requests. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch request. Defaults to 2.
        verbose (bool, optional): Print additional error information. Defaults to False.
        failed_log_path (str, optional): Path to save a log of failed entity retrievals. Defaults to './data/failed_ent_log.txt'.

    Returns:
        None: The function saves the processed triplets to a TXT file.
    """

    if type(file_path) == str:
        entity_list = list(load_to_set(file_path))[:nrows]
    elif type(file_path) == list:
        entity_list = set()
        for file in file_path:
            entity_list.update(load_to_set(file))
        entity_list = list(entity_list)[:nrows]
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'
        
    entity_list_size = len(entity_list)

    client = Client()
    
    failed_ents = []

    with open(output_file_path, 'w') as file:
        pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_entity_triplet, entity_list[i0], client,
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i0 for i0 in range(0, entity_list_size)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Entities Triplets"):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                if result:
                    # Write the result to the file as it is received
                    with open(output_file_path, 'a') as file:
                        for triplet in result:
                            file.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
            except HTTPError as http_err:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"Error: {e}")

        # Save failed entities to a log file
        if failed_ents:
            with open(failed_log_path, 'w') as log_file:
                for ent in failed_ents:
                    log_file.write(f"{ent}\n")

def fetch_entity_details(rdf: str, results: dict, client: Client) -> dict:
    """
    Fetches basic entity details (RDF, title, description, alias, etc.) from Wikidata.

    Args:
        rdf (str): The RDF identifier of the entity.
        results (dict): A dictionary template to store fetched details.
        client (Client): Wikidata client for API requests.

    Returns:
        dict: A dictionary containing the fetched details.
    """
    # Copies results template, fetches data from Wikidata, parses it with BeautifulSoup, and populates the results dictionary.
    
    r = results.copy()
    
    if not rdf and 'Q' != rdf[0]: return r # Return placeholders when RDF is blank
    
    r['RDF'] = rdf
    
    entity = client.get(rdf, load=True)
    if entity.data:
        r['Title'] = entity.label.get('en')
        r['Description'] = entity.description.get('en')
        if entity.id != rdf: r['Forwarding'] = entity.id
        if 'sitelinks' in entity.data.keys() and 'enwiki' in entity.data['sitelinks'].keys():
            r['URL'] = entity.data['sitelinks']['enwiki']['url']
        if 'en' in entity.data['aliases'].keys():
            r['Alias'] = "|".join([ent['value'] for ent in entity.data['aliases']['en']])
    else: return r
    
    try:
        url = f"http://www.wikidata.org/wiki/{rdf}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            r['MDI'] = fetch_freebase_id(soup)
    finally:
        return r

def fetch_freebase_id(soup: BeautifulSoup) -> str:
    """
    Extracts the Freebase ID from the entity's Wikidata page.

    Args:
        soup (BeautifulSoup): Parsed HTML content of the Wikidata page.

    Returns:
        str: The Freebase ID of the entity or an empty string if not found.
    """
    # Extracts and returns the Freebase ID from the parsed HTML content.
    try:
        # Find the div with id 'P646' which presumably contains the Freebase ID
        fb_id_container = soup.find('div', id="P646")
        if fb_id_container:
            # Navigate to the specific 'a' tag that contains the Freebase ID
            fb_id_link = fb_id_container.find('div', class_="wikibase-statementview-mainsnak")
            if fb_id_link:
                fb_id_link = fb_id_link.find('a')
                if fb_id_link and fb_id_link.text:
                    return fb_id_link.text.strip()  # Return the text, stripping any extra whitespace
        return ""  # Return an empty string if any part of the path is not found or the link has no text
    
    except Exception as e:
        return ''

def fetch_entity_triplet(rdf: str, client: Client) -> List[List[str]]:
    """
    Retrieves the triplet relationships an entity has on Wikidata.

    Args:
        rdf (str): The RDF identifier of the entity.
        client (Client): Wikidata client for API requests.

    Returns:
        List[List[str]]: A list of triplets (head, relation, tail) related to the entity.
    """
    
    if not rdf and 'Q' != rdf[0]: return []
    
    entity = client.get(rdf, load=True)
    
    triplets = []
    ent_data = entity.data['claims']
    for e0 in ent_data:
        for e1 in ent_data[e0]:
            if ('datavalue' in set(e1['mainsnak'].keys()) 
                and type(e1['mainsnak']['datavalue']['value']) != str
                and 'id' in set(e1['mainsnak']['datavalue']['value'].keys())
                and 'Q' == e1['mainsnak']['datavalue']['value']['id'][0]):
                triplets.append([entity.id, e0, e1['mainsnak']['datavalue']['value']['id']])
                
            if ('qualifiers' in set(e1.keys())):
                for e2 in e1['qualifiers']:
                    for e3 in e1['qualifiers'][e2]:
                        if ('datavalue' in set(e3.keys())
                            and type(e3['datavalue']['value']) != str
                            and 'id' in set(e3['datavalue']['value'].keys())
                            and 'Q' == e3['datavalue']['value']['id'][0]):
                            triplets.append([entity.id, e2, e3['datavalue']['value']['id']])
    return triplets
    
#------------------------------------------------------------------------------
'Webscrapping for Relationship Info'

def update_relationship_data(rel_df: pd.DataFrame, missing_rels:list, max_workers: int = 10,
                              max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_rel_log.txt') -> None:
    """
    Fetches additional relationship details concurrently for a list of properties from Wikidata and appends them to the provided DataFrame.

    Args:
        rel_df (pd.DataFrame): DataFrame containing existing relationship data.
        missing_rels (list): List of relationship identifiers that need to be fetched.
        max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed fetches. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch. Defaults to 2.
        verbose (bool, optional): Print error details. Defaults to False.
        failed_log_path (str, optional): Path to log failed relationship retrievals. Defaults to './data/failed_rel_log.txt'.

    Returns:
        pd.DataFrame: The combined DataFrame with the newly fetched relationship data appended to the existing data.
    """
    
    rel_list = list(missing_rels)
    rel_list_size = len(rel_list)

    results_template = {
        'Property': '',
        'Title': '',
        'Description': '',
        'Alias': '',
    }
    
    data = []
    failed_props = []

    client = Client()

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_relationship_details, rel_list[i], results_template, client,
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i for i in range(rel_list_size)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching Relationships Data'):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                data.append(result)
            except HTTPError as http_err:
                failed_props.append(rel_list[futures[future]])  # Track the failed entity
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                failed_props.append(rel_list[futures[future]])  # Track the failed property
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                failed_props.append(rel_list[futures[future]])  # Track the failed property
                if verbose: print(f"Error: {e}")
    
    # Save failed properties to a log file
    if failed_props:
        with open(failed_log_path, 'w') as log_file:
            for prop in failed_props:
                log_file.write(f"{prop}\n")
                
    df = pd.DataFrame(data)
    
    df = df.dropna(subset=['Property'])
    df = df[df['Property'].str.strip() != '']  # Then, remove empty strings
    
    combined_df = pd.concat([rel_df, df], ignore_index=True)
    combined_df = sort_by_qid(combined_df, column_name='Property')

    return combined_df


def process_relationship_data(file_path: str, output_file_path: str, nrows: int = None, max_workers: int = 10,
                              max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_rel_log.txt') -> None:
    """
    Fetches relationship details concurrently for a list of properties from Wikidata and saves them to a CSV file.

    Args:
        file_path (str): Path to the input file containing relationship identifiers.
        output_file_path (str): Path to save the processed CSV file.
        nrows (int, optional): Number of rows to process. Defaults to None.
        max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed fetches. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch. Defaults to 2.
        verbose (bool, optional): Print error details. Defaults to False.
        failed_log_path (str, optional): Path to log failed relationship retrievals. Defaults to './data/failed_rel_log.txt'.

    Returns:
        None: The function saves the processed relationship data to a CSV file.
    """
    
    rel_list = list(load_to_set(file_path))[:nrows]
    rel_list_size = len(rel_list)

    results_template = {
        'Property': '',
        'Title': '',
        'Description': '',
        'Alias': '',
    }
    
    data = []
    failed_props = []

    client = Client()

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_relationship_details, rel_list[i], results_template, client,
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i for i in range(rel_list_size)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching Relationships Data'):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                data.append(result)
            except HTTPError as http_err:
                failed_props.append(rel_list[futures[future]])  # Track the failed entity
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                failed_props.append(rel_list[futures[future]])  # Track the failed property
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                failed_props.append(rel_list[futures[future]])  # Track the failed property
                if verbose: print(f"Error: {e}")
    
    # Save failed properties to a log file
    if failed_props:
        with open(failed_log_path, 'w') as log_file:
            for prop in failed_props:
                log_file.write(f"{prop}\n")
                
    df = pd.DataFrame(data)
    
    df = df.dropna(subset=['Property'])
    df = df[df['Property'].str.strip() != '']  # Then, remove empty strings
    
    df = sort_by_qid(df, column_name='Property')

    # Save the updated and sorted DataFrame
    df.to_csv(output_file_path, index=False)
    print("\nData processed and saved to", output_file_path)


def process_relationship_hierarchy(file_path: str, output_file_path: str, nrows: int = None, max_workers: int = 10,
                                  max_retries: int = 3, timeout: int = 2, verbose: bool = False) -> None:
    """
    Scrapes the relationship hierarchy from Wikidata and saves the triplets to a TXT file.

    Args:
        file_path (str): Path to the input file containing relationship identifiers.
        output_file_path (str): Path to save the output TXT file.
        nrows (int, optional): Number of rows to process. Defaults to None.
        max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed fetches. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch. Defaults to 2.
        verbose (bool, optional): Print error details. Defaults to False.

    Returns:
        None: The function saves the relationship hierarchy triplets to a TXT file.
    """
    
    rel_list = list(load_to_set(file_path))[:nrows]
    rel_list_size = len(rel_list)

    client = Client()

    with open(output_file_path, 'w') as file:
        pass

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_relationship_triplet, rel_list[i], client,
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i for i in range(rel_list_size)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching Relationships Hierarchy'):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                if result:
                    # Write the result to the file as it is received
                    with open(output_file_path, 'a') as file:
                        for triplet in result:
                            file.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
            except HTTPError as http_err:
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                if verbose: print(f"Error: {e}") 
            
    print("\nData processed and saved to", output_file_path)

def process_properties_list(property_list_path:str, max_properties: int = 12109, limit: int = 500, max_workers: int = 10,
                            max_retries: int = 3, timeout: int = 2, verbose: bool = False) -> None:
    """
    Fetches basic relationship details (title, description, alias) from Wikidata.

    Args:
        prop (str): The property identifier of the relationship.
        results (dict): A dictionary template to store fetched details.
        client (Client): Wikidata client for API requests.

    Returns:
        dict: A dictionary with the fetched relationship details.
    """
    total_queries = math.ceil(max_properties / limit)
    rel_list = []

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare futures for all offsets
        futures = {executor.submit(retry_fetch, fetch_properties_sublist, i * limit, limit, max_retries = max_retries, timeout = timeout, verbose = verbose): i for i in range(total_queries)}
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching Property Lists'):    
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                rel_list.append(result)
            except TimeoutError:
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                if verbose: print(f"Error: {e}")
    
    rel_list = [item for sublist in rel_list for item in sublist]
    rel_list = sort_qid_list(rel_list)
    
    with open(property_list_path, 'w') as f:
        for property_name in rel_list:
            f.write(f"{property_name}\n")
            
    print("\nData processed and saved to", property_list_path)

def fetch_relationship_details(prop: str, results: dict, client: Client) -> dict:
    """
    Fetches basic details for an entity from Wikidata.
    
    Args:
        prop (str): The Property identifier of the Relationship.
        results (dict): A dictionary template to store the results.
    
    Returns:
        dict: A dictionary with fetched details or placeholders if prop is blank.
    """
    r = results.copy()
    
    if not prop and 'P' != prop[0]: return r # Return placeholders when RDF is blank
    
    rel = client.get(prop, load=True)
    if rel.data:
        r['Property'] = rel.id
        r['Title'] = rel.label.get('en')
        r['Description'] = rel.description.get('en')
        if 'en' in rel.data['aliases'].keys():
            r['Alias'] = "|".join([r0['value'] for r0 in rel.data['aliases']['en']])
    return r

def fetch_properties_sublist(offset: int, limit: int) -> List[str]:
    """
    Fetches a sublist of properties from Wikidata based on the given offset and limit.

    Args:
        offset (int): Starting point for fetching properties.
        limit (int): Number of properties to fetch.

    Returns:
        List[str]: A list of property names.
    """
    url = f'https://www.wikidata.org/w/index.php?title=Special:ListProperties/&limit={limit}&offset={offset}'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        rels = soup.find('ol', class_='special')
        return [rel.get('title').replace('Property:', '') for rel in rels.find_all('a')]
    except Exception as e:
        return []

def fetch_relationship_triplet(prop: str, client: Client) -> List[List[str]]:
    """
    Fetches triplet relationships for a property from Wikidata.

    Args:
        prop (str): Property identifier of the relationship.
        client (Client): Wikidata client for API requests.

    Returns:
        List[List[str]]: A list of triplets (head, relation, tail) related to the property.
    """
    if not prop and 'P' != prop[0]: return []
    
    rel = client.get(prop, load=True)
    rel_data = rel.data['claims']
    triplet = []
    for r0 in rel_data:
        for r1 in rel_data[r0]:
            if ('datavalue' in set(r1['mainsnak'].keys()) 
                and type(r1['mainsnak']['datavalue']['value']) != str 
                and 'id' in set(r1['mainsnak']['datavalue']['value'].keys())
                and 'P' == r1['mainsnak']['datavalue']['value']['id'][0]):
                triplet.append([rel.id, r0, r1['mainsnak']['datavalue']['value']['id']])
    return triplet
    
#------------------------------------------------------------------------------
'MISC'

def search_wikidata_relevant_id(entity_name: str, topk: int = 1) -> str:
    """
    Searches Wikidata for an entity by name and returns the ID of the most relevant result.

    Args:
        entity_name (str): Name of the entity to search for.

    Returns:
        str: The ID of the most relevant entity or an empty string if not found.
    """
    url = "https://www.wikidata.org/w/api.php"

    # Set up the parameters for the API query
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json"
    }

    # Send the GET request to the Wikidata API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        if 'search' in data and len(data['search']) > 0:
            # Return the ID of the most relevant (first) result
            # most_relevant_id = data['search'][0]['id']
            # return most_relevant_id
            most_relevants = []
            for i0 in range(min(topk, len(data['search']))):
                entity = {
                    'Qid': data['search'][i0]['id'],
                    'Title': data['search'][i0]['display']['label']['value'],
                    'Description': data['search'][i0]['display']['description']['value'] if 'description' in data['search'][i0]['display'].keys() else ''
                    }
                most_relevants.append(entity)
            return most_relevants

    return ""

def retry_fetch(func, *args, max_retries=3, timeout=2, verbose=False, **kwargs):
    """
    Retries a function call with a specified timeout and maximum retries.

    Args:
        func: The function to be retried.
        max_retries (int): Maximum number of retries for the function.
        timeout (int): Timeout in seconds for each attempt.
        verbose (bool): Print error messages during retries.

    Returns:
        The function's return value, or raises an exception after retries are exhausted.
    """
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Attempt the function call with a timeout
            return func(*args, **kwargs)
        except HTTPError as http_err:
            last_exception = http_err
            # Check if the error is a 404 and skip retries if true
            if http_err.code == 404:
                if verbose: print(f"HTTP Error 404 for {args[0]}. Skipping further retries.")
                raise last_exception # Return an empty dictionary or handle as appropriate
            else:
                if verbose: print(f"HTTPError on attempt {attempt + 1} for {args[0]}: {http_err}. Retrying...")
        except TimeoutError as timeout_err:
            last_exception = timeout_err
            if verbose: print(f"TimeoutError on attempt {attempt + 1} for {args[0]}. Retrying...")
        except Exception as e:
            last_exception = e
            if verbose: print(f"Error on attempt {attempt + 1} for {args[0]}: {e}. Retrying...")
        time.sleep(1)  # Optional: wait a bit before retrying
    if verbose: print(f"Failed after {max_retries} retries for {args[0]}.")
    
    # Raise the last exception encountered if all retries fail
    if last_exception:
        raise last_exception
    else:
        return {}  # In case there's no specific exception to raise
