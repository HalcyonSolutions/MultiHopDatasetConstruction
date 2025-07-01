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
import threading
from typing import List, Union, Dict, Tuple, Set, DefaultDict

from wikidata.entity import EntityId

from utils.basic import load_to_set, sort_by_qid, sort_qid_list
from utils import sparql_queries


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
            entity_list.update(load_to_set(file_path))
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

def process_entity_triplets(file_path: Union[str, List[str]], output_file_path: str, nrows: int = None, max_workers: int = 10,
                            max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_ent_log.txt') -> None:
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
    
    failed_ents = []
    forward_data = {}

    with open(output_file_path, 'w') as file:
        pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_head_entity_triplets, entity_list[i0],
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i0 for i0 in range(0, entity_list_size)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Entities Triplets"):
            try:
                result, forward_dict, _ = future.result(timeout=timeout)  # Apply timeout here
                if forward_dict: forward_data.update(forward_dict)

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
        
        
        # Convert the dictionary to a pandas DataFrame
        if forward_data:
            forward_data = {k: v for k, v in forward_data.items() if k != v}  # Remove self-references
            if isinstance(file_path, list): 
                forward_path = file_path[0].replace('.txt', '_forwarding.txt')
            else: 
                forward_path = file_path.replace('.txt', '_forwarding.txt')
            
            forward_df = pd.DataFrame(list(forward_data.items()), columns=["QID-to", "QID-from"])
            forward_df = sort_by_qid(forward_df, column_name = 'QID-to')
            forward_df.to_csv(forward_path, index=False)
            print("\nForward data saved to", forward_path)

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

def fetch_head_entity_triplets(qid: str, mode: str="expanded") \
    -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str,str,str],List[str]]]:
    """
    Retrieves the triplet relationships an entity has on Wikidata.

    Args:
        qid (str): The QID identifier of the entity.
        mode (str): Mode of processing triplets. Options are 'expanded', 'separate', 'ignore'.

    Returns:
        Set[Tuple[str, str, str]]: A list of triplets (head, relation, tail) related to the entity
        Dict[str, str]: the forwarding ID if any.
        Dict[Tuple[str,str,str],List[str]]]: The dictionary to recover attributes from triplets

    """
    assert mode in ["expanded", "separate", "ignore"], "Invalid mode for fetch_head_entity_triplets."
    assert qid and 'Q' == qid[0], "Your QID must be prefixed by a Q"

    client = get_thread_local_client()
    entity = client.get(EntityId(qid), load=True)

    triplets: set[tuple[str, str, str]] = set()
    ent_data = entity.data
    
    forward_dict = {}
    if entity.id != qid: forward_dict: Dict[str, str] = {entity.id: qid}

    qualifier_triplets = DefaultDict(list)

    if ent_data is None:
        raise ValueError(f"Entity {qid} not found in Wikidata.")
    ent_claims = ent_data["claims"]
    if not isinstance(ent_claims, dict):
        raise ValueError(f"Entity {qid} is not the expected type (dict).")

    for relation in ent_claims.keys():
        for statement in ent_claims[relation]:
            if ('datavalue' in statement['mainsnak'].keys()
                and isinstance(statement['mainsnak']['datavalue']['value'], dict)
                and 'id' in statement['mainsnak']['datavalue']['value'].keys()
                and 'Q' == statement['mainsnak']['datavalue']['value']['id'][0]):
                # triplets.append([entity.id, e0, e1['mainsnak']['datavalue']['value']['id']])

                triplet = (qid, relation, statement['mainsnak']['datavalue']['value']['id'])

                if not all([isinstance(elem, str) for elem in triplet]):
                    raise ValueError(f"An Element in triplet {triplet} is not a string")

                triplets.add(triplet)
                
                if mode != 'ignore' and ('qualifiers' in statement.keys()):
                    for qual_relation in statement['qualifiers']:
                        for qual_tail in statement['qualifiers'][qual_relation]:
                            if ('datavalue' in qual_tail.keys()
                                and isinstance(qual_tail['datavalue']['value'], dict)
                                and 'id' in qual_tail['datavalue']['value'].keys()
                                and 'Q' == qual_tail['datavalue']['value']['id'][0]):
                                # Legacy behavior. Perhaps incorrect.
                                if mode == 'expanded':
                                    triplets.add((qid, qual_relation, qual_tail['datavalue']['value']['id']))
                                elif mode == 'separate':
                                    qualifier_triplets[triplet].append([qual_relation, qual_tail['datavalue']['value']['id']])

    return triplets, forward_dict, qualifier_triplets

def fetch_tail_entity_triplets_and_qualifiers_optimized(
    qid: str, mode: str = "expanded"
) -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str, str, str], List[Tuple[str, str]]]]:
    """
    Retrieves triplets where an entity is the tail, including their qualifiers,
    using a single optimized SPARQL query.

    Warning: The inherent cross product in the query makes it very costly. 
    Thus it is recommended to run "ignore" for most cases. 
    You can test this expense by running the query on the entity "Q82955".

    Args:
        qid (str): The QID identifier of the entity.
        mode (str): Mode of processing triplets. Options are 'expanded', 'separate', 'ignore'.

    Returns:
        triplets (Set[Tuple[str, str, str]]): A list of triplets (head, relation, tail) related to the entity
        forward_dict (Dict[str, str]): Dictionary that forwards old ids to new ones.
        qualifier_triplets (Dict[Tuple[str,str,str],List[str]]]: The dictionary to recover attributes from triplets
    """
    assert mode in ["expanded", "separate", "ignore"], "Invalid mode for fetch_entity_triplet_as_tail."
    assert qid and qid.startswith('Q'), "Your QID must be prefixed by a Q"

    client = get_thread_local_client()
    entity_obj = client.get(EntityId(qid), load=True) # Handles potential redirects
    entity_id = entity_obj.id # Use the canonical ID

    forward_dict: Dict[str, str] = {}
    if entity_id != qid:
        forward_dict = {qid: entity_id} # Store original: canonical

    triplets: Set[Tuple[str, str, str]] = set()
    # For 'separate' mode: maps main triplet to list of (qual_prop_pid, qual_value_id_or_literal)
    qualifiers_map = DefaultDict(list)

    if mode == "ignore":
        # TODO: Maybe you want to parameterize `fetch_tail_entity_triplets_and_qualifiers_optimized` to take in limit. 
        sparql_query = sparql_queries.TAILS_WITHOUT_QUALIFIERS_COMPLICATED.format(entity_id=entity_id, limit=200)
    else:
        sparql_query = sparql_queries.TAILS_WITH_QUALIFIERS.format(entity_id=entity_id)

    # To group qualifiers by statement if a statement has multiple qualifiers
    # statement_uri -> (main_triplet, list_of_qualifiers)
    temp_statement_data: DefaultDict[tuple, Set[Tuple[str,str]]] = DefaultDict(set)

    # try:
    response = requests.get(url, params=params, timeout=60) # Increased timeout
    response.raise_for_status()
    data = response.json()

    for result in data.get("results", {}).get("bindings", []):
        head_uri = result.get("head_uri", {}).get("value", "")
        relation_uri = result.get("property", {}).get("value", "")
        statement_uri = result.get("statement", {}).get("value", "") # Important for grouping

        head_qid = head_uri.split("/")[-1] if head_uri and head_uri.startswith("http://www.wikidata.org/entity/Q") else None
        relation_pid = relation_uri.split("/")[-1] if relation_uri and relation_uri.startswith("http://www.wikidata.org/entity/P") else None # property URI from statementProperty

        if not (head_qid and relation_pid and statement_uri):
            continue

        main_triplet = (head_qid, relation_pid, entity_id) # entity_id is the fixed tail

        # Add main triplet to the set (duplicates won't be added due to set properties)
        triplets.add(main_triplet)

        if mode == 'ignore':
            continue

        qual_property_uri = result.get("qual_property_uri", {}).get("value", "")
        qual_value_raw = result.get("qual_value", {}).get("value", "")
        qual_value_is_item_str = result.get("qual_v_is_item", {}).get("value", "false")
        qual_value_is_item = qual_value_is_item_str.lower() == "true"
        
        # For now we dont care about non-item qualifiers
        if not qual_value_is_item:
            continue

        if qual_property_uri and qual_value_raw:
            qual_property_pid = qual_property_uri.split("/")[-1] if qual_property_uri.startswith("http://www.wikidata.org/entity/P") else None # Qualifier properties are PIDs (entities)

            if not qual_property_pid: # Should be a P-entity
                qual_property_pid = qual_property_uri.split("/")[-1] if "http://www.wikidata.org/prop/P" in qual_property_uri else None # for wikibase:qualifier direct props
                assert qual_property_pid is not None, "I actually never expected this to be the case"

            # Final qualifier value - already a QID string or literal string
            # TODO: Need to check on this bad boi.
            qual_value_processed = qual_value_raw

            # Apply original filtering logic for qualifiers if needed
            # Your original code checked if qual_prop starts with P and qual_value starts with Q
            if qual_property_pid and qual_property_pid.startswith("P"):
                # If you ONLY want qualifiers where the VALUE is an ITEM (QID)
                # if not (qual_value_is_item and qual_value_processed.startswith("Q")):
                # continue # or handle as needed
                current_qualifier_pair = (qual_property_pid, qual_value_processed)
                temp_statement_data[main_triplet].add(current_qualifier_pair)
                        


    # Post-process temp_statement_data for final output structures
    for stmt_main_triplet, stmt_qualifiers_list in temp_statement_data.items():
        if mode == 'expanded':
            for qual_p, qual_v in stmt_qualifiers_list:
                # The 'expanded' mode semantic needs clarification.
                # A qualifier (qual_p, qual_v) modifies the stmt_main_triplet.
                # Adding (head_qid_of_main_triplet, qual_p, qual_v) might be misleading.
                # A common expanded form for qualifiers is (statement_id, qual_p, qual_v),
                # or reifying the statement.
                # Here, we'll add it as (head_main, qual_p, qual_v) as per implied original logic
                triplets.add((stmt_main_triplet[0], qual_p, qual_v))
        elif mode == 'separate':
            qualifiers_map[stmt_main_triplet].extend(stmt_qualifiers_list)


    # except requests.exceptions.RequestException as e:
    #     print(f"Error during SPARQL query for {qid} (entity_id: {entity_id}): {e}")
    #     if hasattr(e, 'response') and e.response is not None:
    #         print(f"Response content: {e.response.text}")
    # except ValueError as e: # For JSON decoding errors
    #     print(f"Error decoding JSON response for {qid} (entity_id: {entity_id}): {e}")

    # If mode is 'separate', ensure lists in qualifiers_map are unique if necessary (already handled by check during append)
    # If you didn't check for duplicates when appending to temp_statement_data's list:
    # if mode == 'separate':
    # for key in qualifiers_map:
    # qualifiers_map[key] = list(set(qualifiers_map[key]))

    final_qualifiers_map = qualifiers_map if mode == 'separate' else DefaultDict(list)
    return triplets, forward_dict, final_qualifiers_map

def fetch_tail_entity_triplets(qid: str) \
    -> Tuple[Set[Tuple[str, str, str]], Dict[str, str]]:
    """
    Retrieves the triplet relationships where an entity is the tail (object) on Wikidata.
    This function uses the Wikidata API to find all entities that reference the given entity.

    Args:
        qid (str): The QID identifier of the entity to find as a tail.

    Returns:
        Set[Tuple[str, str, str]]: A list of triplets (head, relation, tail) where the given entity is the tail
        Dict[str, str]: the forwarding ID if any.

    TODO:
        Maybe setup qualifiers here as well ?
    """
    assert qid and 'Q' == qid[0], "Your QID must be prefixed by a Q"

    client = get_thread_local_client()
    entity = client.get(EntityId(qid), load=True)
    
    forward_dict = {}
    if entity.id != qid: forward_dict: Dict[str, str] = {entity.id: qid}
    
    # Use the actual entity ID for the query
    entity_id = entity.id
    
    # Initialize the set of triplets
    triplets: set[tuple[str, str, str]] = set()
    
    # Construct the SPARQL query to find all entities that reference this entity
    sparql_query = f"""
    SELECT ?item ?itemLabel ?property ?propertyLabel WHERE {{
      ?item ?property wd:{entity_id} .
      ?item wdt:P31 ?type .  # Only include items that have a type
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    # Execute the query using the Wikidata Query Service
    url = "https://query.wikidata.org/sparql"
    params = {
        "query": sparql_query,
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        
        # Process the results
        for result in data.get("results", {}).get("bindings", []):
            head_uri = result.get("item", {}).get("value", "")
            relation_uri = result.get("property", {}).get("value", "")
            
            # Extract the QID and PID from the URIs
            head_qid = head_uri.split("/")[-1] if head_uri else ""
            # head_qid = head_qid.split("-")[0] if head_qid else ""
            relation_pid = relation_uri.split("/")[-1] if relation_uri else ""
            
            # Only add valid triplets
            if head_qid and relation_pid and head_qid.startswith("Q") and relation_pid.startswith("P"):
                triplets.add((head_qid, relation_pid, entity_id))
    
    except Exception as e:
        print(f"Error fetching triplets where {qid} is tail: {e}")
    
    return triplets, forward_dict

def fetch_entity_triplet_bidirectional(qid: str, mode: str="expanded") \
    -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str,str,str],List[str]]]:
    """
    Retrieves all triplet relationships where an entity appears as either head or tail on Wikidata.

    Args:
        qid (str): The QID identifier of the entity.
        mode (str): Mode of processing triplets. Options are 'expanded', 'separate', 'ignore'.

    Returns:
        all_triplets (Set[Tuple[str, str, str]]): A list of triplets (head, relation, tail) where either `head` or `tail` are `qid`
        forward_Dict (Dict[str, str]): the forwarding ID if any.
        merged_qualifier_triplets (Dict[Tuple[str,str,str],List[str]]]): The dictionary to recover attributes from triplets
    """
    assert mode in ["expanded", "separate", "ignore"], "Invalid mode for fetch_entity_triplet_bidirectional."
    assert qid and 'Q' == qid[0], "Your QID must be prefixed by a Q"
    
    # Get triplets where entity is head
    head_triplets, forward_dict, head_qualifier_triplets = fetch_head_entity_triplets(qid, mode)
    tail_triplets, tail_forward_dict, tail_qualifier_triplets = fetch_tail_entity_triplets_and_qualifiers_optimized(qid, mode)

    
    # Merge the forward dictionaries
    forward_dict.update(tail_forward_dict)
    
    # Merge the triplets
    all_triplets = head_triplets.union(tail_triplets)
    
    # Merge the qualifier triplets dictionaries
    merged_qualifier_triplets = DefaultDict(list)
    for triplet, qualifiers in head_qualifier_triplets.items():
        merged_qualifier_triplets[triplet].extend(qualifiers)
    for triplet, qualifiers in tail_qualifier_triplets.items():
        merged_qualifier_triplets[triplet].extend(qualifiers)
    
    return all_triplets, forward_dict, merged_qualifier_triplets
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

    client = get_thread_local_client()
    rel = client.get(prop, load=True)
    if rel.data:
        # r['Property'] = rel.id
        r['Property'] = prop
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
