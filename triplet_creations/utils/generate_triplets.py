# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:50:10 2024

@author: Eduin Hernandez

Summary: Utils for generating triplets
"""
from tqdm import tqdm
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from utils.wikidata import fetch_details_qid
from utils.basic import load_to_set

def _create_triplets(head:str , rel_tail: dict) -> list:
    """
    Creates triplets from the head entity and related tails.
    
    Args:
        head (str): The RDF identifier of the head entity.
        rel_tail (dict): A dictionary with relationships as keys and lists of related tail entities as values.
    
    Returns:
        list: A list of triplets [head, relationship, tail].
    """
    return [[head, k0, k1] for k0 in rel_tail.keys() for k1 in rel_tail[k0] if ((k1 is not None) and ('Q' in str(k1)))]

def _create_triplets_valid(head:str , rel_tail: dict, valid_set: set) -> list:
    """
    Creates triplets from the head entity and related tails, but only if the tail is in the valid set.
    
    Args:
        head (str): The RDF identifier of the head entity.
        rel_tail (dict): A dictionary with relationships as keys and lists of related tail entities as values.
        valid_set (set): A set of str containing all valid entities that can be used as tails.
    
    Returns:
        list: A list of triplets [head, relationship, tail].
    """
    return [[head, k0, k1] for k0 in rel_tail.keys() for k1 in rel_tail[k0] if (k1 in valid_set)]

def fetch_and_create_triplets(rdf: str, results: dict, p_map: dict) -> list:
    """
    Fetches details for an RDF entity and creates triplets.
    
    Args:
        rdf (str): The RDF identifier of the entity.
        results (dict): A dictionary template to store the results.
        p_map (dict): A mapping of property keys to property IDs.
    
    Returns:
        list: A list of created triplets for the entity.
    """
    if rdf:
        res = fetch_details_qid(rdf, results, p_map)
        return _create_triplets(rdf, res)
    return []

def fetch_and_create_triplets_valid(rdf: str, results: dict, p_map: dict, valid_set: set) -> list:
    """
    Fetches details for an RDF entity and creates triplets, but only if the tail is in the valid set.
    
    Args:
        rdf (str): The RDF identifier of the entity.
        results (dict): A dictionary template to store the results.
        p_map (dict): A mapping of property keys to property IDs.
        valid_set (set): A set of str containing all valid entities that can be used as tails.
    
    Returns:
        list: A list of created triplets for the entity.
    """
    if rdf:
        res = fetch_details_qid(rdf, results, p_map)
        return _create_triplets(rdf, res)
    return []
    
def generate_triplets(file_path:str, output_file_path:str, p_map: dict, nrows:int = None) -> None:
    """
    Processes the input data to fetch details and create triplets, then saves the triplets to a file.
    
    Args:
        file_path (str): The path to the input CSV file containing RDF entities.
        output_file_path (str): The path to save the output triplets file.
        p_map (dict): A mapping of property keys to property IDs.
        nrows (int): Optional, for testing only a subsamples of entries.
    
    Returns:
        None
    """
    # Use dictionary comprehension to create dictionary entries for each property key
    results = {key: {} for key in p_map.keys()}
    
    entity_list = list(load_to_set(file_path))[:nrows]
    entity_list_size = len(entity_list)
    with open(output_file_path, 'w') as file:
        pass
        
    with open(output_file_path, 'w') as file:
        for i0 in tqdm(range(0, entity_list_size), total=entity_list_size, desc="Fetching data"):
            triplets = fetch_and_create_triplets(entity_list[i0], results, p_map)
                
            for triplet in triplets:
                file.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
    
    print("\nData processed and saved to", output_file_path)
    
def generate_triplets_valid(file_path:str, output_file_path:str, p_map: dict, valid_path: str, nrows:int = None) -> None:
    """
    Processes the input data to fetch details and create triplets, but only if the tail is in the valid set, then saves the triplets to a file.
    
    Args:
        file_path (str): The path to the input CSV file containing RDF entities.
        output_file_path (str): The path to save the output triplets file.
        p_map (dict): A mapping of property keys to property IDs.
        valid_path (str): A set of str containing all valid entities that can be used as tails.
        nrows (int): Optional, for testing only a subsamples of entries.
    
    Returns:
        None
    """
    # Use dictionary comprehension to create dictionary entries for each property key
    results = {key: {} for key in p_map.keys()}
    
    entity_list = list(load_to_set(file_path))[:nrows]
    entity_list_size = len(entity_list)
    valid_set = load_to_set(valid_path)
    with open(output_file_path, 'w') as file:
        pass
        
    with open(output_file_path, 'w') as file:
        for i0 in tqdm(range(0, entity_list_size), total=entity_list_size, desc="Fetching data"):
            triplets = fetch_and_create_triplets_valid(entity_list[i0], results, p_map, valid_set)
                
            for triplet in triplets:
                file.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
    
    print("\nData processed and saved to", output_file_path)

def _triplet_writer_thread(output_file_path, queue):
    with open(output_file_path, 'a') as file:
        while True:
            triplets = queue.get()
            if triplets is None:
                break
            for triplet in triplets:
                file.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
            queue.task_done()

def generate_triplets_threaded(file_path:str, output_file_path:str, p_map: dict, nrows:int = None, max_workers: int = 4) -> None:
    """
    Uses threads to processes the input data to fetch details and create triplets, then saves the triplets to a file.
    
    Args:
        file_path (str): The path to the input CSV file containing RDF entities.
        output_file_path (str): The path to save the output triplets file.
        p_map (dict): A mapping of property keys to property IDs.
        nrows (int): Optional, for testing only a subsamples of entries.
        max_workers (int): Optional, for selecting the number of threads to work with.
    
    Returns:
        None
    """
    # Use dictionary comprehension to create dictionary entries for each property key
    results = {key: {} for key in p_map.keys()}
    
    entity_list = list(load_to_set(file_path))[:nrows]
    entity_list_size = len(entity_list)
    
    queue = Queue()
    writer = Thread(target=_triplet_writer_thread, args=(output_file_path, queue))
    writer.start()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_create_triplets, entity_list[i0], results, p_map) for i0 in range(0, entity_list_size)]
        
        for future in tqdm(futures, total=len(futures), desc="Fetching data"):
            triplets = future.result()
            if triplets:
                queue.put(triplets)
    
    queue.put(None)
    writer.join()
    
    print("\nData processed and saved to", output_file_path)
    
def generate_triplets_valid_threaded(file_path:str, output_file_path:str, p_map: dict, valid_path: str, nrows:int = None, max_workers: int = 4) -> None:  
    """
    Uses threads to processes the input data to fetch details and create triplets, but only if the tail is in the valid set, then saves the triplets to a file.
    
    Args:
        file_path (str): The path to the input CSV file containing RDF entities.
        output_file_path (str): The path to save the output triplets file.
        p_map (dict): A mapping of property keys to property IDs.
        valid_path (str): A set of str containing all valid entities that can be used as tails.
        nrows (int): Optional, for testing only a subsamples of entries.
        max_workers (int): Optional, for selecting the number of threads to work with.
    
    Returns:
        None
    """
    
    # Use dictionary comprehension to create dictionary entries for each property key
    results = {key: {} for key in p_map.keys()}
    
    entity_list = list(load_to_set(file_path))[:nrows]
    entity_list_size = len(entity_list)
    valid_set = load_to_set(valid_path)
    
    queue = Queue()
    writer = Thread(target=_triplet_writer_thread, args=(output_file_path, queue))
    writer.start()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_create_triplets_valid, entity_list[i0], results, p_map, valid_set) for i0 in range(0, entity_list_size)]
        
        for future in tqdm(futures, total=len(futures), desc="Fetching data"):
            triplets = future.result()
            if triplets:
                queue.put(triplets)
    
    queue.put(None)
    writer.join()
    
    print("\nData processed and saved to", output_file_path)