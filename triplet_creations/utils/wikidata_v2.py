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
import os

import math
import pandas as pd

import time
import requests
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from wikidata.client import Client

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
from typing import List, Union, Dict, Tuple

from utils.basic import load_to_set, sort_by_qid, sort_qid_list


# Create a thread-local storage object
thread_local = threading.local()

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
        'QID': '',
        'Title': '',
        'Description': '',
        'Alias': '',
        'MID': '',
        'URL': '',
        'Forwarding': '',
    }
    
    data = []
    failed_entities = []

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_entity_details, entity_list[i], results_template,
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
    
    df = df.dropna(subset=['QID'])
    df = df[df['QID'].str.strip() != '']  # Then, remove empty strings

    missing = set(entity_list) - set(df['QID'].tolist())
    
    # Create a DataFrame from the set with empty values for other columns
    new_rows = pd.DataFrame([[qid] + ['']*(len(results_template)-1) for qid in missing], columns=df.columns)
    
    # Append new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    combined_df = pd.concat([entity_df, df], ignore_index=True)
    combined_df.drop_duplicates(subset='QID', inplace=True)
    
    # Sort the DataFrame by the "QID" column
    combined_df = sort_by_qid(combined_df, column_name = 'QID')

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
    
    if isinstance(file_path, str):
        entity_list = list(load_to_set(file_path))[:nrows]
    elif isinstance(file_path, list):
        entity_list = set()
        for file in file_path:
            entity_list.update(load_to_set(file))
        entity_list = list(entity_list)[:nrows]
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'
        
    entity_list_size = len(entity_list)

    results_template = {
        'QID': '',
        'Title': '',
        'Description': '',
        'Alias': '',
        'MID': '',
        'URL': '',
        'Forwarding': '',
    }

    data = []
    failed_ents = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_entity_details, entity_list[i0], results_template,
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
    
    missing = set(entity_list) - set(df['QID'].tolist())
    
    # Create a DataFrame from the set with empty values for other columns
    new_rows = pd.DataFrame([[qid] + ['']*(len(results_template)-1) for qid in missing], columns=df.columns)
    
    # Append new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    
    df.drop_duplicates(subset='QID', inplace=True)
    
    # Sort the DataFrame by the "QID" column
    df = sort_by_qid(df, column_name = 'QID')
    
    # Save the updated and sorted DataFrame
    df.to_csv(output_file_path, index=False)
    print("\nData processed and saved to", output_file_path)

def process_entity_triplets(file_path: Union[str, List[str]], triplet_file_path: str, qualifier_file_path: str, forwarding_file_path: str,
                            nrows: int = None, max_workers: int = 10,
                            max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_ent_log.txt') -> None:
    """
    Scrapes and processes triplet relationships for a set of entities from Wikidata and saves the data to a TXT file.

    Args:
        file_path (str or list): Path to the input file or list of files with entity IDs.
        triplet_file_path (str): Path to save the processed triplets.
        qualifier_file_path (str): Path to save the processed qualifiers.
        forwarding_file_path (str): Path to save the forwarding entities.
        nrows (int, optional): Number of rows to process (None for all). Defaults to None.
        max_workers (int, optional): Maximum number of threads for parallel processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed fetch requests. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch request. Defaults to 2.
        verbose (bool, optional): Print additional error information. Defaults to False.
        failed_log_path (str, optional): Path to save a log of failed entity retrievals. Defaults to './data/failed_ent_log.txt'.

    Returns:
        None: The function saves the processed triplets to a TXT file.
    """

    if isinstance(file_path, str):
        entity_list = list(load_to_set(file_path))[:nrows]
    elif isinstance(file_path, list):
        entity_list = set()
        for file in file_path:
            entity_list.update(load_to_set(file))
        entity_list = list(entity_list)[:nrows]
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'
    
    # Verify if the paths are openable and writable

    if not os.path.exists(triplet_file_path):
        open(triplet_file_path, 'w').close()
    if not os.path.exists(qualifier_file_path):
        open(qualifier_file_path, 'w').close()
    if not os.path.exists(forwarding_file_path):
        open(forwarding_file_path, 'w').close()
    
    assert os.access(triplet_file_path, os.W_OK), 'Error! The triplet_file_path is not writable'
    assert os.access(qualifier_file_path, os.W_OK), 'Error! The qualifier_file_path is not writable'
    assert os.access(forwarding_file_path, os.W_OK), 'Error! The forwarding_file_path is not writable'

    entity_list_size = len(entity_list)
    
    failed_ents = []
    forward_data = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_entity_triplet, entity_list[i0],
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i0 for i0 in range(0, entity_list_size)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Entities Triplets"):
            try:
                facts_result, qualifier_result, forward_dict = future.result(timeout=timeout)  # Apply timeout here
                if forward_dict: forward_data.update(forward_dict)

                if facts_result:
                    # Write the result to the file as it is received
                    with open(triplet_file_path, 'a') as file:
                        for triplet in facts_result:
                            file.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
                if qualifier_result:
                    # Write the result to the file as it is received
                    with open(qualifier_file_path, 'a') as file:
                        for quintent in qualifier_result:
                            file.write(f"{quintent[0]}\t{quintent[1]}\t{quintent[2]}\t{quintent[3]}\t{quintent[4]}\n")
            except HTTPError as http_err:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                failed_ents.append(entity_list[futures[future]])  # Track the failed entity
                if verbose: print(f"Error: {e}")
        
        
        # Convert the dictionary to a pandas DataFrame
        if forward_data:
            forward_data = {k: v for k, v in forward_data.items() if k != v}  # Remove self-references
            forward_df = pd.DataFrame(list(forward_data.items()), columns=["QID-to", "QID-from"])
            forward_df = sort_by_qid(forward_df, column_name = 'QID-to')
            forward_df.to_csv(forwarding_file_path, index=False)
            print("\nForward data saved to", forwarding_file_path)

        # Save failed entities to a log file
        if failed_ents:
            with open(failed_log_path, 'w') as log_file:
                for ent in failed_ents:
                    log_file.write(f"{ent}\n")

def process_entity_forwarding(file_path: Union[str, List[str]], output_file_path: str, nrows: int = None, max_workers: int = 10,
                            max_retries: int = 3, timeout: int = 2, verbose: bool = False) -> None:

    """
    Fetches forwarding entity IDs for a list of entities from Wikidata and saves the results to a CSV file.
    """
    if isinstance(file_path, str):
        entity_list = list(load_to_set(file_path))[:nrows]
    elif isinstance(file_path, list):
        entity_list = set()
        for file in file_path:
            entity_list.update(load_to_set(file))
        entity_list = list(entity_list)[:nrows]
    else:
        assert False, 'Error! The file_path must either be a string or a list of strings'
    
    entity_list_size = len(entity_list)

    # start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_entity_forwarding, entity_list[i0],
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i0 for i0 in range(0, entity_list_size)}

        data = {}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Forwarding Entities"):
            try:
                result = future.result(timeout=timeout)  # Apply timeout here
                data.update(result)
            except HTTPError as http_err:
                if verbose: print(f"HTTPError: {http_err}")
            except TimeoutError:
                if verbose: print("TimeoutError: Task took too long and was skipped.")
            except Exception as e:
                if verbose: print(f"Error: {e}")

        data = {k: v for k, v in data.items() if k != v}  # Remove self-references

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Time taken to fetch forwarding entities: {elapsed_time:.2f} seconds")

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(list(data.items()), columns=["QID-to", "QID-from"])

    # Sort the DataFrame by the "QID" column
    df = sort_by_qid(df, column_name = 'QID-to')
    
    # Save the updated and sorted DataFrame
    df.to_csv(output_file_path, index=False)
    print("\nData processed and saved to", output_file_path)


def fetch_entity_details(qid: str, results: dict) -> dict:
    """
    Fetches basic entity details (QID, title, description, alias, etc.) from Wikidata.

    Args:
        qid (str): The QID identifier of the entity.
        results (dict): A dictionary template to store fetched details.

    Returns:
        dict: A dictionary containing the fetched details.
    """
    # Copies results template, fetches data from Wikidata, parses it with BeautifulSoup, and populates the results dictionary.
    
    r = results.copy()
    
    if not qid and 'Q' != qid[0]: return r # Return placeholders when QID is blank
    
    r['QID'] = qid

    client = get_thread_local_client()
    entity = client.get(qid, load=True)
    if entity.data:
        r['Title'] = entity.label.get('en')
        r['Description'] = entity.description.get('en')
        if entity.id != qid: r['Forwarding'] = entity.id
        if 'sitelinks' in entity.data.keys() and 'enwiki' in entity.data['sitelinks'].keys():
            r['URL'] = entity.data['sitelinks']['enwiki']['url']
        if 'en' in entity.data['aliases'].keys():
            r['Alias'] = "|".join([ent['value'] for ent in entity.data['aliases']['en']])
    else: return r
    
    try:
        url = f"http://www.wikidata.org/wiki/{qid}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            r['MID'] = fetch_freebase_id(soup)
    finally:
        return r

def fetch_entity_forwarding(qid: str) -> str:
    """
    Fetches the forwarding entity ID from Wikidata.

    Args:
        qid (str): The QID identifier of the entity.

    Returns:
        str: The forwarding entity ID or an empty string if not found.
    """
    
    if not qid and 'Q' != qid[0]: return {}
    
    client = get_thread_local_client()
    entity = client.get(qid, load=True)
    
    if entity.data:
        if entity.id != qid: return {entity.id: qid}
    
    return {}

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

def fetch_entity_triplet(qid: str) -> Tuple[List[List[str]], Dict[str, str]]:
    """
    Retrieves the triplet relationships an entity has on Wikidata.

    Args:
        qid (str): The QID identifier of the entity.

    Returns:
        List[List[str]]: A list of triplets (head, relation, tail) related to the entity
        Dict[str, str]: the forwarding ID if any.
    """
    
    if not qid and 'Q' != qid[0]: return []

    client = get_thread_local_client()
    entity = client.get(qid, load=True)
    
    triplets = []
    qualifiers = []
    ent_data = entity.data['claims']

    forward_dict = {}
    if entity.id != qid: forward_dict = {entity.id: qid}

    for e0 in ent_data:
        for e1 in ent_data[e0]:
            if ('datavalue' in e1['mainsnak'] 
                and isinstance(e1['mainsnak']['datavalue']['value'], dict)
                and 'id' in e1['mainsnak']['datavalue']['value']
                and 'Q' == e1['mainsnak']['datavalue']['value']['id'][0]):
                # triplets.append([entity.id, e0, e1['mainsnak']['datavalue']['value']['id']])
                triplets.append([qid, e0, e1['mainsnak']['datavalue']['value']['id']])
                
                # Qualifiers should only be considered if the fact exists
                if ('qualifiers' in e1): 
                    for e2 in e1['qualifiers']:
                        for e3 in e1['qualifiers'][e2]:
                            if ('datavalue' in e3
                                and isinstance(e3['datavalue']['value'], dict)
                                and 'id' in e3['datavalue']['value']
                                and 'Q' == e3['datavalue']['value']['id'][0]):
                                # triplets.append([entity.id, e2, e3['datavalue']['value']['id']])
                                qualifiers.append([qid, e0, e1['mainsnak']['datavalue']['value']['id'],
                                e2, e3['datavalue']['value']['id']])

    return triplets, qualifiers, forward_dict
    
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

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_relationship_details, rel_list[i], results_template,
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

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_relationship_details, rel_list[i], results_template,
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

    with open(output_file_path, 'w') as file:
        pass

    # Using ThreadPoolExecutor to fetch data concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submitting all tasks to the executor
        futures = {executor.submit(retry_fetch, fetch_relationship_triplet, rel_list[i],
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

def fetch_relationship_details(prop: str, results: dict) -> dict:
    """
    Fetches basic details for an entity from Wikidata.
    
    Args:
        prop (str): The Property identifier of the Relationship.
        results (dict): A dictionary template to store the results.
    
    Returns:
        dict: A dictionary with fetched details or placeholders if prop is blank.
    """
    r = results.copy()
    
    if not prop and 'P' != prop[0]: return r # Return placeholders when QID is blank

    r['Property'] = prop

    client = get_thread_local_client()
    rel = client.get(prop, load=True)
    if rel.data:
        # r['Property'] = rel.id
        r['Title'] = rel.label.get('en')
        r['Description'] = rel.description.get('en')
        r['Forwarding'] = rel.id if rel.id != prop else ''
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

def fetch_relationship_triplet(prop: str) -> List[List[str]]:
    """
    Fetches triplet relationships for a property from Wikidata.

    Args:
        prop (str): Property identifier of the relationship.

    Returns:
        List[List[str]]: A list of triplets (head, relation, tail) related to the property.
    """
    if not prop and 'P' != prop[0]: return []

    client = get_thread_local_client()
    rel = client.get(prop, load=True)
    rel_data = rel.data['claims']
    triplet = []
    for r0 in rel_data:
        for r1 in rel_data[r0]:
            if ('datavalue' in set(r1['mainsnak'].keys()) 
                and isinstance(r1['mainsnak']['datavalue']['value'], dict) 
                and 'id' in set(r1['mainsnak']['datavalue']['value'].keys())
                and 'P' == r1['mainsnak']['datavalue']['value']['id'][0]):
                # triplet.append([rel.id, r0, r1['mainsnak']['datavalue']['value']['id']])
                triplet.append([prop, r0, r1['mainsnak']['datavalue']['value']['id']])
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
                    'QID': data['search'][i0]['id'],
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

def get_thread_local_client() -> Client:
    """
    Returns a thread-local instance of the Wikidata client.
    """

    if not hasattr(thread_local, "client"):
        thread_local.client = Client()
    return thread_local.client