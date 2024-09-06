# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:59:44 2024

@author: Eduin Hernandez
"""
import pandas as pd

import requests
from bs4 import BeautifulSoup

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.basic import load_to_set, sort_by_qid

#------------------------------------------------------------------------------
'Webscrapping for Entity Info'
def fetch_entity_details(entity: str, template: dict) -> dict:
    """
    Fetches base details for a given entity from Wikidata.

    Args:
        entity (str): The RDF identifier of the entity.
        template (dict): A dictionary template to store the results.

    Returns:
        dict: A dictionary with fetched details or placeholders if the entity is blank.
    """
    if entity:
        return fetch_base_details(entity, template)
    return template.copy()  # Handle cases where RDF is blank

def process_entity_data(file_path: str, output_file_path: str, nrows: int = None, max_workers: int = 10) -> None:
    """
    Processes entity data by fetching details and saving them to a CSV file.
    
    Args:
        input_path (str): The path to the input file containing entity identifiers.
        output_path (str): The path to save the output CSV file.
        nrows (int, optional): The number of rows to read from the input file. Defaults to None.
        max_workers (int, optional): The maximum number of workers for threading. Defaults to 10.
    
    Returns:
        None
    """

    entity_list = list(load_to_set(file_path))[:nrows]
    entity_list_size = len(entity_list)

    results_template = {
        'Title': '',
        'Description': '',
        'MDI': '',
        'URL': '',
        'Alias': '',
        'RDF': '',
    }

    data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_entity_details, entity_list[i0], results_template): i0 for i0 in range(0, entity_list_size)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching data"):
            res = future.result()
            data.append(res.copy())

    df = pd.DataFrame(data)
    
    # Sort the DataFrame by the "RDF" column
    df = sort_by_qid(df, column_name = 'RDF')
    
    # Save the updated and sorted DataFrame
    df.to_csv(output_file_path, index=False)
    print("\nData processed and saved to", output_file_path)


def fetch_base_details(rdf: str, results: dict) -> dict:
    """
    Fetches basic details for an entity from Wikidata.
    
    Args:
        rdf (str): The RDF identifier of the entity.
        results (dict): A dictionary template to store the results.
    
    Returns:
        dict: A dictionary with fetched details or placeholders if RDF is blank.
    """
    # Copies results template, fetches data from Wikidata, parses it with BeautifulSoup, and populates the results dictionary.
    
    r = results.copy()
    
    if not rdf:
        return r # Return placeholders when RDF is blank
    
    try:
        url = f"http://www.wikidata.org/wiki/{rdf}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            r['RDF'] = rdf 
            r['Title'] = fetch_title(soup)
            r['Description'] = fetch_description(soup)
            r['MDI'] = fetch_freebase_id(soup)
            r['URL'] = fetch_wiki_url(soup)
            r['Alias'] = fetch_alias(soup)

            return r
        else:
            raise ConnectionError(f"HTTP status code {response.status_code}")
    except Exception as e:
        return r

def fetch_title(soup: BeautifulSoup) -> str:
    """
    Extracts the title of the entity from the Wikidata page.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
    
    Returns:
        str: The title of the entity.
    """
    # Extracts and returns the title from the parsed HTML content.
    try:
        # Extract title
        title_tag = soup.find("title")
        return title_tag.text.split(' - Wikidata')[0] if title_tag else ""
    except Exception as e:
        return ''
    
def fetch_description(soup: BeautifulSoup) -> str:
    """
    Extracts the description of the entity from the Wikidata page.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
    
    Returns:
        str: The description of the entity.
    """
    # Extracts and returns the description from the parsed HTML content.
    try:
        # Extract English description
        description_tag = soup.find('div', class_='wikibase-entitytermsview-heading-description')
        return description_tag.text if description_tag else ""
    except Exception as e:
        return ''
    
def fetch_freebase_id(soup: BeautifulSoup) -> str:
    """
    Extracts the Freebase ID of the entity from the Wikidata page.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
    
    Returns:
        str: The Freebase ID of the entity.
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

def fetch_wiki_url(soup: BeautifulSoup) -> str:
    """
    Extracts the Wikipedia URL of the entity from the Wikidata page.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
    
    Returns:
        str: The Wikipedia URL of the entity.
    """
    # Extracts and returns the Wikipedia URL from the parsed HTML content.
    try:
        url_span = soup.find('span', class_="wikibase-sitelinkview-link wikibase-sitelinkview-link-enwiki")
        if url_span:
            link = url_span.find('a')
            return link.get('href') if link else ''
        return ""  # Return an empty string if any part of the path is not found or the link has no text
    
    except Exception as e:
        return ''

def fetch_alias(soup: BeautifulSoup) -> str:
    """
    Extracts the aliases of the entity from the Wikidata page.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
    
    Returns:
        str: A string containing aliases of the entity separated by '|'.
    """
    # Extracts and returns the aliases from the parsed HTML content.
    try:
        res = ''
        alias_ul = soup.find('ul', class_="wikibase-entitytermsview-aliases")
        for alias_val in alias_ul.find_all('li', class_="wikibase-entitytermsview-aliases-alias"):
            res += alias_val.text + '|'
        if res: res = res[:-1]
        return res  # Return an empty string if any part of the path is not found or the link has no text
    except Exception as e:
        return ''
    
def fetch_property(soup: BeautifulSoup, id_val: str) -> dict:
    """
    Extracts properties of the entity from the Wikidata page by property ID.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
        id_val (str): The ID of the property to extract.
    
    Returns:
        dict: A dictionary of instance types and their QIDs.
    """
    # Extracts and returns properties by ID from the parsed HTML content.
    try:
        prop_portion = soup.find('div', id=id_val)
        
        # Initialize a dictionary to hold the instance types and their QIDs
        res = {}
        # Iterate over each instance type in the 'instance of' div
        for value_snak in prop_portion.find_all('div', class_="wikibase-statementview-mainsnak"):
            link = value_snak.find('a')
            if link:
                qid = link.get('title')  # Get the QID from the title attribute
                instance_type = link.text.strip()  # Get the instance type text
                res[instance_type] = qid
        return res
    except Exception as e:
        return {}

def fetch_property_qid(soup: BeautifulSoup, id_val: str) -> list:
    """
    Extracts QIDs of properties of the entity from the Wikidata page by property ID.

    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
        id_val (str): The ID of the property to extract.

    Returns:
        list: A list of QIDs for the specified property.
    """
    # Extracts and returns a list of QIDs by property ID from the parsed HTML content.

    try:
        prop_portion = soup.find('div', id=id_val)
        
        # Initialize a dictionary to hold the instance types and their QIDs
        res = []
        # Iterate over each instance type in the 'instance of' div
        for value_snak in prop_portion.find_all('div', class_="wikibase-statementview-mainsnak"):
            link = value_snak.find('a')
            if link: res.append(link.get('title'))  # Get the QID from the title attribute
        return res
    except Exception as e:
        return []


def fetch_details(rdf: str, results: dict, p_map: dict) -> dict:
    """
    Fetches detailed properties for an entity from Wikidata.
    
    Args:
        rdf (str): The RDF identifier of the entity.
        results (dict): A dictionary template to store the results.
        p_map (dict): A mapping of property keys to property IDs.
    
    Returns:
        dict: A dictionary with fetched detailed properties.
    """
    # Fetches and returns detailed properties based on the given mapping.

    r = results.copy()
    
    if not rdf:
        return r # Return placeholders when RDF is blank
    try:
        url = f"http://www.wikidata.org/wiki/{rdf}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for k0 in p_map.keys():
                r[k0] = fetch_property(soup, p_map[k0])
            return r
        else:
            raise ConnectionError(f"HTTP status code {response.status_code}")
    except Exception as e:
        return r
    
def fetch_details_qid(rdf: str, results: dict, p_map: dict) -> dict:
    """
    Fetches detailed properties QIDs for an entity from Wikidata.
    
    Args:
        rdf (str): The RDF identifier of the entity.
        results (dict): A dictionary template to store the results.
        p_map (dict): A mapping of property keys to property IDs.
    
    Returns:
        dict: A dictionary with fetched detailed property QIDs.
    """
    # Fetches and returns detailed property QIDs based on the given mapping.

    r = results.copy()
    
    if not rdf:
        return r # Return placeholders when RDF is blank
    try:
        url = f"http://www.wikidata.org/wiki/{rdf}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for k0 in p_map.keys():
                r[k0] = fetch_property_qid(soup, k0)
            return r
        else:
            raise ConnectionError(f"HTTP status code {response.status_code}")
    except Exception as e:
        return r

#------------------------------------------------------------------------------
'Webscrapping for Relationship'
def fetch_relation_details(rdf: str) -> dict:
    """
    Fetches basic details for a relationship from Wikidata.
    
    Args:
        rdf (str): The Propertity identifier of the entity.
    
    Returns:
        dict: A dictionary with fetched details or placeholders if RDF is blank.
    """
    # Copies results template, fetches data from Wikidata, parses it with BeautifulSoup, and populates the results dictionary.
    
    r = {'Property': '',
         'Title': '',
         'Description': '',
         'Alias': ''}
    
    if not rdf:
        return r # Return placeholders when RDF is blank
    
    try:
        url = f"http://www.wikidata.org/wiki/Property:{rdf}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            r['Property'] = rdf 
            r['Title'] = fetch_title(soup)
            r['Description'] = fetch_description(soup)
            r['Alias'] = fetch_alias(soup)

            return r
        else:
            raise ConnectionError(f"HTTP status code {response.status_code}")
    except Exception as e:
        return r

def fetch_inverse_relation(rdf: str) -> str:
    """
    Fetches inverse and subproperty relations for a given RDF property from Wikidata.
    
    Args:
        rdf (str): The RDF identifier of the property.
    
    Returns:
        dict: A dictionary with inverse and subproperty relations.
    """
    # Fetches and returns inverse and subproperty relations for the given RDF property.

    url = f"http://www.wikidata.org/wiki/Property:{rdf}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    inv_dict = fetch_property_relation(soup, "P1696")
    sub_dict = fetch_property_relation(soup, "P1647")
    return inv_dict, sub_dict


def fetch_property_relation(soup: BeautifulSoup, id_val: str) -> dict:
    """
    Extracts property relations of the entity from the Wikidata page by property ID.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content of the Wikidata page.
        id_val (str): The ID of the property relation to extract.
    
    Returns:
        dict: A dictionary of property relations and their QIDs.
    """
    # Extracts and returns property relations by ID from the parsed HTML content.

    relation_div = soup.find('div', id=id_val)
    if not relation_div:
        return {}
    
    relation_container = relation_div.find('div', class_="wikibase-statementview-mainsnak-container")
    if not relation_container:
        return {}
    
    relation_mainsnak = relation_container.find('div', class_='wikibase-statementview-mainsnak')
    if not relation_mainsnak:
        return {}
    
    relation_link = relation_mainsnak.find('a', title=True)
    if not relation_link:
        return {}
    
    relation_id = relation_link.get('title').replace('Property:', '')
    return {relation_id: relation_link.text.strip()}