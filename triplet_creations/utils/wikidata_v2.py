# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:18:44 2024

Summary
-------
Utilities to scrape, process, and retrieve Wikidata entity and relationship
information, with built-in concurrency and robust networking that complies
with Wikimedia’s User-Agent and rate-limit guidelines.

Wikimedia-compliant networking
------------------------------
- User-Agent is loaded from ./configs/config_wiki.ini (section [Wikimedia])
  using required keys: project, repo, mail (optional: version). No defaults.
- The same UA is applied everywhere:
  * a custom urllib opener passed to wikidata.client.Client to avoid 403s and
    preserve headers across redirects,
  * a thread-local requests.Session with retries/backoff that honors
    Retry-After,
  * SPARQLWrapper via the `agent` parameter; queries also include `maxlag`.
- Functions favor short timeouts, exponential backoff with jitter, and
  friendly behavior toward WDQS/Wikidata replicas.
- For more info, see: https://phabricator.wikimedia.org/T400119

Core capabilities
-----------------
- Entity data: labels, descriptions, aliases, sitelinks, Freebase MID, and
  redirect/forwarding resolution.
- Triplets: head, tail, and bidirectional extraction with optional qualifier
  handling ('expanded' | 'separate' | 'ignore'); optimized SPARQL path for
  “tail” relations.
- Relationship (property) data: titles, descriptions, aliases; hierarchy
  scraping and full property listing.
- Search: lightweight `wbsearchentities` helper for name→QID lookup.
- Bulk processors: threaded updaters that read IDs, fetch details/triplets,
  write CSV/TXT, and keep stable QID ordering.
- Resilience: shared retry utility, failure logs, and thread-local clients/
  sessions to reuse connections safely.

Key entry points (non-exhaustive)
---------------------------------
- Networking/session: get_thread_local_client(), get_thread_local_session(),
  get_sparql()
- Entity: fetch_entity_details(), fetch_entity_forwarding(),
  update_entity_data(), process_entity_data()
- Triplets: fetch_head_entity_triplets(), fetch_tail_entity_triplets(),
  fetch_entity_triplet_bidirectional(), process_entity_triplets()
- Relationships: fetch_relationship_details(), update_relationship_data(),
  process_relationship_data(), process_relationship_hierarchy(),
  process_properties_list()
- Search: search_wikidata_relevant_id()
"""

# =============================================================================
# Imports
# =============================================================================

# --- Standard library ---
import os
import math
import random
import re
import time
import threading
import configparser
from pathlib import Path
from typing import List, Union, Dict, Tuple, Set, DefaultDict
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# --- Third-party ---
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from wikidata.client import Client
from wikidata.entity import EntityId
from SPARQLWrapper import SPARQLWrapper, JSON
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib.request

# --- Local ---
from utils.basic import load_to_set, sort_by_qid, sort_qid_list
from utils import sparql_queries

# =============================================================================
# Globals & Configuration (User-Agent)
# =============================================================================

# Create a thread-local storage object
thread_local = threading.local()

# Global rate limiter for Wikidata API requests
class WikidataRateLimiter:
    """
    Thread-safe rate limiter for Wikidata API requests.
    Ensures we don't exceed 10 requests per second as per Wikidata's guidelines.
    """
    def __init__(self, max_requests_per_second: int = 10):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second  # Minimum time between requests
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """
        Wait if necessary to respect the rate limit.
        This method is thread-safe and ensures requests are spaced appropriately.
        """
        with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.min_interval:
                sleep_time = self.min_interval - time_since_last_request
                print(f"Time limiting for {sleep_time} seconds since last request was {time_since_last_request}")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


def load_wikimedia_ua_from_config() -> str:
    """
    Reads ./configs/config_wiki.ini and composes the User-Agent string.
    Requires:
      [Wikimedia]
      project = ...
      repo    = ...
      mail    = ...
    Optional:
      version = ...   # if present, we'll include "project/version"
    """
    cfg_path = Path("./configs/config_wiki.ini")
    if not cfg_path.is_file():
        raise RuntimeError(
            f"config_wiki.ini not found at: {cfg_path}\n"
        )

    cp = configparser.ConfigParser()
    read_ok = cp.read(cfg_path, encoding="utf-8")
    if not read_ok:
        raise RuntimeError(f"Failed to read config file at: {cfg_path}")

    if not cp.has_section("Wikimedia"):
        raise RuntimeError("Missing [Wikimedia] section in config_wiki.ini")

    required = ["project", "repo", "mail"]
    missing = [k for k in required if not cp.has_option("Wikimedia", k) or not cp.get("Wikimedia", k).strip()]
    if missing:
        raise RuntimeError(f"Missing required keys in [Wikimedia]: {', '.join(missing)}")

    project = cp.get("Wikimedia", "project").strip()
    repo    = cp.get("Wikimedia", "repo").strip()
    mail    = cp.get("Wikimedia", "mail").strip()
    version = cp.get("Wikimedia", "version", fallback="").strip()

    if version:
        return f"{project}/{version} (+{repo}; mailto:{mail})"
    return f"{project} (+{repo}; mailto:{mail})"

# Public constant used by sessions/SPARQL
WIKIMEDIA_UA = load_wikimedia_ua_from_config()

# =============================================================================
# HTTP / Client / SPARQL helpers
# =============================================================================

def _build_wikimedia_opener(user_agent: str) -> urllib.request.OpenerDirector:
    class _UAHTTPRedirectHandler(urllib.request.HTTPRedirectHandler):
        # ensure headers (esp. User-Agent) survive redirects
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            new = super().redirect_request(req, fp, code, msg, headers, newurl)
            if new is not None:
                # carry over the original headers to the redirected request
                for k, v in req.header_items():
                    # add_unredirected_header avoids some header stripping
                    new.add_unredirected_header(k, v)
            return new

    opener = urllib.request.build_opener(_UAHTTPRedirectHandler)
    # default headers applied to every request made via this opener
    opener.addheaders = [
        ("User-Agent", user_agent),
        ("Accept", "application/json"),
    ]
    return opener


def get_thread_local_client() -> Client:
    """
    Returns a thread-local Wikidata Client wired to a urllib opener
    that sets a Wikimedia-compliant User-Agent on every request.
    """
    if not hasattr(thread_local, "client"):
        ua = WIKIMEDIA_UA  # already loaded from config_wiki.ini
        opener = _build_wikimedia_opener(ua)
        c = Client(opener=opener)  # <-- key change: pass opener here
        thread_local.client = c
    return thread_local.client


def get_thread_local_session():
    """
    Thread-local requests.Session configured for Wikimedia/Wikidata.
    Includes a compliant User-Agent, retries with backoff, and honors Retry-After.
    """
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        s.headers.update({
            "User-Agent": WIKIMEDIA_UA,
            # (Optional but helpful) explicit Accept to avoid surprises.
            "Accept": "application/json"
        })
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"],
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        thread_local.session = s
    return thread_local.session


def get_sparql(endpoint: str = "https://query.wikidata.org/sparql") -> SPARQLWrapper:
    sparql = SPARQLWrapper(endpoint, agent=WIKIMEDIA_UA)
    sparql.setReturnFormat(JSON)
    # (Optional) be replica-lag friendly for big queries:
    sparql.addParameter("maxlag", "5")
    return sparql

# =============================================================================
# Generic Utilities (retry, error shim)
# =============================================================================

def _raise_urllib_http_error(url: str, code: int, msg: str):
    raise HTTPError(url, code, msg, None, None)


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

    # Start uniformly at random between 0 and 3 seconds
    next_timeout_wait = random.uniform(0, 3)
    time.sleep(next_timeout_wait)
    
    last_exception = None
    
    next_timeout_wait = 5 # For exponential Backoff
    jitter_size = 4
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

        jitter = random.random() * jitter_size - (jitter_size / 2)
        timeout_wait = max(next_timeout_wait + jitter, 1) # Just in cas ethe jitter drops me below 0
        print(f"Sleeping for {timeout_wait} seconds")
        time.sleep(timeout_wait)  # Optional: wait a bit before retrying
        next_timeout_wait = next_timeout_wait * 5  

    if verbose: 
        print(f"Failed after {max_retries} retries for {args[0]}.")
    
    # Raise the last exception encountered if all retries fail
    if last_exception:
        raise last_exception
    else:
        return {}  # In case there's no specific exception to raise

# =============================================================================
# Search & Lookup (API-based)
# =============================================================================

def search_wikidata_relevant_id(entity_name: str, topk: int = 1):
    """
    Searches Wikidata for an entity by name and returns up to 'topk' relevant hits.
    Return: list of dicts [{QID, Title, Description}, ...]; [] if none.
    Raises urllib.error.HTTPError for HTTP errors (to match your retry_fetch).
    """
    if not entity_name:
        return []

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",   # search language
        "uselang": "en",    # response labels/descriptions language
        "type": "item",     # items (not properties)
        "limit": max(1, int(topk)),
        "format": "json",
        "maxlag": "5",      # be nice to the replicas
    }

    sess = get_thread_local_session()
    try:
        response = sess.get(url, params=params, timeout=15)
    except requests.RequestException as e:
        # Map network errors to a 503-ish HTTPError so your retry_fetch can catch them
        _raise_urllib_http_error(url, 503, f"Network error: {e}")

    # Helpful debug:
    # print(f"Searching for entity: {entity_name} with URL: {response.url}")
    # print(f"Response Status Code: {response.status_code}")

    # If rate-limited or temporarily unavailable and server gives Retry-After, do a final short pause
    if response.status_code in (429, 503):
        ra = response.headers.get("Retry-After")
        if ra:
            try:
                time.sleep(min(int(ra), 10))
            except ValueError:
                time.sleep(2)

    # Try decode JSON, but be resilient
    try:
        data = response.json()
    except ValueError:
        data = {}

    # MediaWiki may return 200 with an "error" object (e.g., maxlag)
    if isinstance(data, dict) and "error" in data:
        code = data["error"].get("code", "")
        info = data["error"].get("info", "")
        # Politely back off for maxlag; return empty so caller can retry via retry_fetch
        if code == "maxlag":
            time.sleep(2)
            return []
        _raise_urllib_http_error(response.url, 500, f"API error: {code} {info}")

    # Transport-level errors
    if response.status_code == 403:
        _raise_urllib_http_error(response.url, 403, "Forbidden (likely missing/non-compliant User-Agent or blocked).")
    if response.status_code == 429:
        _raise_urllib_http_error(response.url, 429, "Too many requests (rate-limited).")
    if response.status_code >= 500:
        _raise_urllib_http_error(response.url, response.status_code, f"Server error: {response.text[:200]}")

    # Success path
    hits = data.get("search") or []
    out = []
    for hit in hits[:max(1, int(topk))]:
        display = hit.get("display") or {}
        label = (display.get("label") or {}).get("value") or hit.get("label") or ""
        desc  = (display.get("description") or {}).get("value") or hit.get("description") or ""
        out.append({
            "QID": hit.get("id", ""),
            "Title": label,
            "Description": desc,
        })
    return out


# =============================================================================
# HTML Parsing Helpers (BeautifulSoup)
# =============================================================================

def fetch_freebase_id(source: Union[Dict, BeautifulSoup]) -> str:
    """
    Return the Freebase MID (P646) from an entity JSON 'claims' dict.
    Accepts a BeautifulSoup as a deprecated fallback (returns '').

    Preferred usage:
        fetch_freebase_id(entity.data)

    Args:
        source: entity.data (dict) or a BeautifulSoup (deprecated).

    Returns:
        str: Freebase MID like '/m/02mjmr' or '' if not present.
    """
    # Preferred: entity.data (dict)
    if isinstance(source, dict):
        claims = source.get("claims", {})
        statements = claims.get("P646") or []
        for stmt in statements:
            mainsnak = stmt.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            val = datavalue.get("value")
            # For P646, the value is a string (e.g., "/m/02mjmr")
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    # Deprecated fallback: BeautifulSoup HTML parse (avoid using under new policy)
    # Kept only for compatibility if something else still calls with soup.
    if isinstance(source, BeautifulSoup):
        return ""  # Explicitly do nothing in the HTML path

    return ""

# =============================================================================
# Entity Data: details, forwarding, bulk processing
# =============================================================================

def fetch_entity_details(qid: str, results: dict = {}) -> dict:
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
    entity = client.get(EntityId(qid), load=True)
    if entity.data:
        r['Title'] = entity.label.get('en')
        r['Description'] = entity.description.get('en')
        if entity.id != qid: r['Forwarding'] = entity.id
        if 'sitelinks' in entity.data.keys() and 'enwiki' in entity.data['sitelinks'].keys():
            r['URL'] = entity.data['sitelinks']['enwiki']['url']
        if 'en' in entity.data['aliases'].keys():
            r['Alias'] = "|".join([ent['value'] for ent in entity.data['aliases']['en']])
        r['MID'] = fetch_freebase_id(entity.data)
    
    return r


def fetch_entity_details_batch(qids: List[str], rate_limiter: WikidataRateLimiter, batch_size: int = 50, timeout: int = 30) -> List[dict]:
    """
    Fetches basic entity details for multiple entities in batches using SPARQL queries.
    This is more efficient than individual requests as it reduces the number of API calls.

    Args:
        qids (List[str]): List of QID identifiers of the entities.
        batch_size (int, optional): Number of entities to query per batch. Defaults to 50.
        timeout (int, optional): Timeout in seconds for each SPARQL query. Defaults to 30.
        rate_limiter (WikidataRateLimiter): Rate limiter instance to use. Otherwise wikidata sad

    Returns:
        List[dict]: A list of dictionaries containing the fetched details for each entity.
    """
    results = []
    
    # Process entities in batches
    for i in range(0, len(qids), batch_size):
        batch_qids = qids[i:i + batch_size]
        
        # Filter out invalid QIDs
        valid_qids = [qid for qid in batch_qids if qid and qid.startswith('Q')]
        
        if not valid_qids:
            # Add empty results for invalid QIDs
            for qid in batch_qids:
                results.append({
                    'QID': qid,
                    'Title': '',
                    'Description': '',
                    'Alias': '',
                    'MID': '',
                    'URL': '',
                })
            continue
        
        # Create VALUES clause for SPARQL query
        values_clause = " ".join([f"wd:{qid}" for qid in valid_qids])
        
        sparql_query = f"""
        SELECT ?entity ?entityLabel ?entityDescription ?entityAltLabel ?freebaseId ?enwikiUrl WHERE {{
          VALUES ?entity {{ {values_clause} }}
          
          OPTIONAL {{ ?entity wdt:P646 ?freebaseId . }}
          OPTIONAL {{ 
            ?enwikiSitelink schema:about ?entity ;
                           schema:isPartOf <https://en.wikipedia.org/> ;
                           schema:name ?enwikiTitle .
            BIND(CONCAT("https://en.wikipedia.org/wiki/", ENCODE_FOR_URI(?enwikiTitle)) AS ?enwikiUrl)
          }}
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en" . 
            ?entity rdfs:label ?entityLabel .
            ?entity schema:description ?entityDescription .
            ?entity skos:altLabel ?entityAltLabel .
          }}
        }}
        """
        
        try:
            # Apply rate limiting before making SPARQL request
            rate_limiter.wait_if_needed()
            
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(timeout)
            
            query_results = sparql.query()
            sparql_results = query_results.convert()
            
            # Create a mapping from QID to result data
            entity_data = {}
            for result in sparql_results["results"]["bindings"]:
                entity_uri = result.get("entity", {}).get("value", "")
                qid = entity_uri.split("/")[-1] if entity_uri else ""
                
                if qid not in entity_data:
                    entity_data[qid] = {
                        'QID': qid,
                        'Title': result.get("entityLabel", {}).get("value", ""),
                        'Description': result.get("entityDescription", {}).get("value", ""),
                        'Alias': '',
                        'MID': result.get("freebaseId", {}).get("value", ""),
                        'URL': result.get("enwikiUrl", {}).get("value", ""),
                    }
                
                # Handle multiple aliases
                alt_label = result.get("entityAltLabel", {}).get("value", "")
                if alt_label and alt_label not in entity_data[qid]['Alias']:
                    if entity_data[qid]['Alias']:
                        entity_data[qid]['Alias'] += "|" + alt_label
                    else:
                        entity_data[qid]['Alias'] = alt_label
            
            # Add results for all QIDs in this batch (including those not found)
            for qid in valid_qids:
                if qid in entity_data:
                    results.append(entity_data[qid])
                else:
                    # Entity not found, add empty result
                    results.append({
                        'QID': qid,
                        'Title': '',
                        'Description': '',
                        'Alias': '',
                        'MID': '',
                        'URL': '',
                    })
                    
        except Exception as e:
            print(f"Error in batch SPARQL query: {e}")
            # Add empty results for this batch
            for qid in valid_qids:
                results.append({
                    'QID': qid,
                    'Title': '',
                    'Description': '',
                    'Alias': '',
                    'MID': '',
                    'URL': '',
                })
    
    return results


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
    entity = client.get(EntityId(qid), load=True)
    
    if entity.data:
        if entity.id != qid: return {entity.id: qid}
    
    return {}


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


def process_entity_data(
    entity_list: List[str],
    max_workers: int = 10,
    max_retries: int = 3,
    timeout: int = 2,
    verbose: bool = False,
    failed_log_path: str = "./data/failed_ent_log.txt",
) -> pd.DataFrame:
    """
    Processes entity data by fetching details from Wikidata for each entity and saving the results to a CSV file.

    Args:
        file_path (str or list): Path to the file or a list of files containing entity IDs.
        output_file_path (str): Path to save the processed CSV file.
        max_workers (int, optional): Maximum number of threads for parallel processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed requests. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each fetch request. Defaults to 2.
        verbose (bool, optional): Print additional error information. Defaults to False.
        failed_log_path (str, optional): Path to save a log of failed entity retrievals. Defaults to './data/failed_ent_log.txt'.

    Returns:
        None: The function saves the processed data to a CSV file.
    """
        
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
    
    return df


def process_data_batch_generic(
    id_list: List[str],
    is_entity: bool = True, # Otherwise relation
    batch_size: int = 50,
    max_workers: int = 10,
    max_retries: int = 3,
    timeout: int = 30,
    verbose: bool = False,
    failed_log_path: str = "./data/failed_log.txt",
) -> pd.DataFrame:
    """
    Generic function to process either entity or relationship data by fetching details from Wikidata 
    in batches using SPARQL queries with threading.

    Args:
        id_list (List[str]): List of IDs to process (QIDs for entities, PIDs for relationships).
        is_entity (bool, optional): True for entities, False for relationships. Defaults to True.
        batch_size (int, optional): Number of items to query per batch. Defaults to 50.
        max_workers (int, optional): Maximum number of threads for parallel processing. Defaults to 10.
        max_retries (int, optional): Maximum number of retries for failed requests. Defaults to 3.
        timeout (int, optional): Timeout in seconds for each SPARQL query. Defaults to 30.
        verbose (bool, optional): Print additional error information. Defaults to False.
        failed_log_path (str, optional): Path to save a log of failed retrievals. Defaults to './data/failed_log.txt'.

    Returns:
        pd.DataFrame: DataFrame containing the processed data.
    """
    
    data = []
    failed_ids = []
    
    # Create local rate limiter for this batch processing session
    # NOTE: Do not move into the `with ThreadPoolExecutor...` scope.
    local_rate_limiter = WikidataRateLimiter(max_requests_per_second=10)
    
    # Create batches
    num_batches = math.ceil(len(id_list) / batch_size)
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(id_list))
        batches.append(id_list[start_idx:end_idx])
    
    # Determine the appropriate description for progress bar
    desc = "Fetching Entity Data (Batched)" if is_entity else "Fetching Relationship Data (Batched)"
    
    # Process batches using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        funct_to_call = fetch_entity_details_batch if is_entity else fetch_relationship_details_batch
        futures = {
            executor.submit(
                retry_fetch, 
                funct_to_call, 
                ids_batch, 
                local_rate_limiter,
                batch_size, 
                timeout,
                max_retries=max_retries, 
                timeout=timeout, 
                verbose=verbose
            ): i for i, ids_batch in enumerate(batches)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                batch_results = future.result(timeout=timeout)
                data.extend(batch_results)
            except HTTPError as http_err:
                batch_idx = futures[future]
                failed_ids.extend(batches[batch_idx])
                if verbose: 
                    print(f"HTTPError in batch {batch_idx + 1}: {http_err}")
            except TimeoutError:
                batch_idx = futures[future]
                failed_ids.extend(batches[batch_idx])
                if verbose: 
                    print(f"TimeoutError in batch {batch_idx + 1}: Task took too long and was skipped.")
            except Exception as e:
                batch_idx = futures[future]
                failed_ids.extend(batches[batch_idx])
                if verbose: 
                    print(f"Error in batch {batch_idx + 1}: {e}")
    
    # Save failed IDs to a log file
    if failed_ids:
        with open(failed_log_path, 'w') as log_file:
            for id_item in failed_ids:
                log_file.write(f"{id_item}\n")
        if verbose:
            print(f"Saved {len(failed_ids)} failed items to {failed_log_path}")
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Determine the appropriate column name for deduplication and sorting
        id_column = 'QID' if is_entity else 'Property'
        df.drop_duplicates(subset=id_column, inplace=True)
        # Sort the DataFrame by the appropriate ID column
        df = sort_by_qid(df, column_name=id_column)
    
    return df


# =============================================================================
# Triplets (head/tail/bidirectional)
# =============================================================================

def fetch_head_entity_triplets(qid: str, limit: int = None, mode: str="expanded") \
    -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str,str,str],List[str]]]:
    """
    Retrieves the triplet relationships an entity has on Wikidata.

    Args:
        qid (str): The QID identifier of the entity.
        limit (int): Limit on the amount of triplets to obtain for the given qid. 
        mode (str): Mode of processing triplets. Options are 'expanded', 'separate', 'ignore'.

    Returns:
        Set[Tuple[str, str, str]]: A list of triplets (head, relation, tail) related to the entity
        Dict[str, str]: the forwarding ID if any.
        Dict[Tuple[str,str,str],List[str]]]: The dictionary to recover attributes from triplets

    """
    assert mode in ["expanded", "separate", "ignore"], "Invalid mode for fetch_head_entity_triplets."
    assert qid and 'Q' == qid[0], "Your QID must be prefixed by a Q"

    if limit is None: limit = math.inf

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
            if len(triplets) > limit: break
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
        if len(triplets) > limit: break

    return triplets, forward_dict, qualifier_triplets


def fetch_tail_entity_triplets(
    qid: str, limit: int, mode: str = "expanded"
) -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str, str, str], List[Tuple[str, str]]]]:
    """
    Retrieves triplets where an entity is the tail, including their qualifiers,
    using a single optimized SPARQL query.

    Warning: The inherent cross product in the query makes it very costly. 
    Thus it is recommended to run "ignore" for most cases. 
    You can test this expense by running the query on the entity "Q82955".

    Args:
        qid (str): The QID identifier of the entity.
        limit (int): Max amount of triplets to return per given `qid`
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
        sparql_query = sparql_queries.TAILS_WITHOUT_QUALIFIERS_COMPLICATED.format(entity_id=entity_id, limit=limit)
    else:
        sparql_query = sparql_queries.TAILS_WITH_QUALIFIERS.format(entity_id=entity_id)

    # To group qualifiers by statement if a statement has multiple qualifiers
    # statement_uri -> (main_triplet, list_of_qualifiers)
    temp_statement_data: DefaultDict[tuple, Set[Tuple[str,str]]] = DefaultDict(set)

    spq = SPARQLWrapper("https://query.wikidata.org/sparql")
    spq.setQuery(sparql_query)
    spq.setReturnFormat(JSON)
    spq.setMethod("POST")
    spq.setTimeout(60)
    query_results = spq.query()
    results = query_results.convert()

    for result in results["results"]["bindings"]: # type: ignore
        assert isinstance(result, Dict)
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


def fetch_tail_entity_triplets_old(qid: str) \
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


def describe_fetch_entity_triplet_bidirectional(qid: str, limit=100) \
    -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str,str,str],List[str]]]:
    """
    Retrieves all triplet relationships where an entity appears as either head or tail on Wikidata.
    Warning: Will not do qualifiers for now.
    Note: This was an attempt to use a native operation in wikidata to retrieve all data about a triplet.
    But for those that are very dense in tails and heads this `Describe` operation takes ages. 
    So its pretty useless at scaling. But I leave here in case its useful for some other case.

    Args:
        qid (str): The QID identifier of the entity.
        limit (int): Limit of how many entities per direction to query for.

    Returns:
        triplets (Set[Tuple[str, str, str]]): A list of triplets (head, relation, tail) related to the entity
        forward_dict (Dict[str,str]): Simple remapping of old id into new id according to wikidata.
        qualifiers_triplets Dict[Tuple[str,str,str],List[str]]]: The dictionary to recover qualifiers from triplets
    Warning: 
        Currently not recovering qualifiers
    """
    # NOTE: qualifiers are not being recovered at the moment.
    #  Thats because it requires a lot more compute that is not worth at the moment

    assert qid and 'Q' == qid[0], "Your QID must be prefixed by a Q"

    q_pattern = re.compile(r'http://www.wikidata.org/entity/(Q\d+$)')
    p_pattern = re.compile(r'http://www.wikidata.org/prop/direct/(P\d+$)')

    # Get Forwarding Dict
    client = get_thread_local_client()
    entity = client.get(EntityId(qid), load=True)
    forward_dict = {}
    if entity.id != qid:
        forward_dict: Dict[str, str] = {entity.id: qid}
    new_qid = entity.id

    # Form the payload
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = { 'User-Agent': 'HumbleScraperBot' }
    payload = {"query": f"DESCRIBE wd:{new_qid}", "format": "json"}
    
    # Receive Results
    r = requests.get(endpoint_url, params=payload, headers=headers)
    results = r.json()

    tail_triplets = set()
    head_triplets = set()
    for result in results["results"]["bindings"]:   
        subject_val = q_pattern.search(result["subject"]["value"])
        predicate_val = p_pattern.search(result["predicate"]["value"])
        object_val = q_pattern.search(result["object"]["value"])

        if subject_val is None \
           or predicate_val is None \
           or object_val is None:
            continue

        if new_qid == subject_val:
            if len(head_triplets) <= limit:
                head_triplets.add((subject_val.group(1), predicate_val.group(1), object_val.group(1)))
        elif new_qid ==object_val:
            if len(tail_triplets) <= limit:
                tail_triplets.add((subject_val.group(1), predicate_val.group(1), object_val.group(1)))

    triplets = set()
    triplets.update(tail_triplets)
    triplets.update(head_triplets)
    qualifiers_triplets = {}
    return triplets, forward_dict, qualifiers_triplets


def fetch_entity_triplet_bidirectional(qid: str, limit: int, mode: str="expanded") \
    -> Tuple[Set[Tuple[str, str, str]], Dict[str, str], Dict[Tuple[str,str,str],List[str]]]:
    """
    Retrieves all triplet relationships where an entity appears as either head or tail on Wikidata.

    Args:
        qid (str): The QID identifier of the entity.
        limit (int): Max amount of triplets to return per given `qid`
        mode (str): Mode of processing triplets. Options are 'expanded', 'separate', 'ignore'.

    Returns:
        all_triplets (Set[Tuple[str, str, str]]): A list of triplets (head, relation, tail) where either `head` or `tail` are `qid`
        forward_Dict (Dict[str, str]): the forwarding ID if any.
        merged_qualifier_triplets (Dict[Tuple[str,str,str],List[str]]]): The dictionary to recover attributes from triplets
    """
    assert mode in ["expanded", "separate", "ignore"], "Invalid mode for fetch_entity_triplet_bidirectional."
    assert qid and 'Q' == qid[0], "Your QID must be prefixed by a Q"
    
    # Get triplets where entity is head
    head_triplets, forward_dict, head_qualifier_triplets = fetch_head_entity_triplets(qid, limit, mode)
    tail_triplets, tail_forward_dict, tail_qualifier_triplets = fetch_tail_entity_triplets(qid, limit, mode)
    
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


def process_entity_triplets(file_path: Union[str, List[str]], triplet_file_path: str, forwarding_file_path: str, nrows: int = None, max_workers: int = 10,
                            max_retries: int = 3, timeout: int = 2, verbose: bool = False, failed_log_path: str = './data/failed_ent_log.txt') -> None:
    """
    Scrapes and processes triplet relationships for a set of entities from Wikidata and saves the data to a TXT file.

    Args:
        file_path (str or list): Path to the input file or list of files with entity IDs.
        triplet_file_path (str): Path to save the processed triplets.
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
    if not os.path.exists(forwarding_file_path):
        open(forwarding_file_path, 'w').close()
    
    assert os.access(triplet_file_path, os.W_OK), 'Error! The triplet_file_path is not writable'
    assert os.access(forwarding_file_path, os.W_OK), 'Error! The forwarding_file_path is not writable'

    entity_list_size = len(entity_list)
    
    failed_ents = []
    forward_data = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(retry_fetch, fetch_head_entity_triplets, entity_list[i0],
                                   max_retries = max_retries, timeout = timeout, verbose = verbose): i0 for i0 in range(0, entity_list_size)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Entities Triplets"):
            try:
                facts_result, forward_dict, _ = future.result(timeout=timeout)  # Apply timeout here
                if forward_dict: forward_data.update(forward_dict)

                if facts_result:
                    # Write the result to the file as it is received
                    with open(triplet_file_path, 'a') as file:
                        for triplet in facts_result:
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
            forward_df = pd.DataFrame(list(forward_data.items()), columns=["QID-to", "QID-from"])
            forward_df = sort_by_qid(forward_df, column_name = 'QID-to')
            forward_df.to_csv(forwarding_file_path, index=False)
            print("\nForward data saved to", forwarding_file_path)

        # Save failed entities to a log file
        if failed_ents:
            with open(failed_log_path, 'w') as log_file:
                for ent in failed_ents:
                    log_file.write(f"{ent}\n")


# =============================================================================
# Relationships: details, hierarchy, property lists
# =============================================================================

def fetch_relationship_details(prop: str, results: dict = {}) -> dict:
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
    rel = client.get(EntityId(prop), load=True)
    if rel.data:
        r['Title'] = rel.label.get('en')
        r['Description'] = rel.description.get('en')
        if 'en' in rel.data['aliases'].keys():
            r['Alias'] = "|".join([r0['value'] for r0 in rel.data['aliases']['en']])
    return r

def fetch_relationship_details_batch(pids: List[str], rate_limiter: WikidataRateLimiter, batch_size: int = 50, timeout: int = 30) -> List[dict]:
    """
    Fetches basic relationship details for multiple properties in batches using SPARQL queries.
    This is more efficient than individual requests as it reduces the number of API calls.

    Args:
        pids (List[str]): List of PID identifiers of the properties.
        rate_limiter (WikidataRateLimiter): Rate limiter instance to use.
        batch_size (int, optional): Number of properties to query per batch. Defaults to 50.
        timeout (int, optional): Timeout in seconds for each SPARQL query. Defaults to 30.

    Returns:
        List[dict]: A list of dictionaries containing the fetched details for each property.
    """
    results = []
    # Useful later
    empty_results = {'Property': '', 'Title': '', 'Description': '', 'Alias': ''}
    
    # Process properties in batches
    for i in range(0, len(pids), batch_size):
        batch_pids = pids[i:i + batch_size]
        
        # Filter out invalid PIDs
        valid_pids = [pid for pid in batch_pids if pid and pid.startswith('P')]
        
        if not valid_pids:
            # Add empty results for invalid PIDs
            for pid in batch_pids:
                results.append(empty_results)
            continue
        
        # Create VALUES clause for SPARQL query
        values_clause = " ".join([f"wd:{pid}" for pid in valid_pids])
        
        sparql_query = f"""
        SELECT ?property ?propertyLabel ?propertyDescription ?propertyAltLabel WHERE {{
          VALUES ?property {{ {values_clause} }}
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en" . 
            ?property rdfs:label ?propertyLabel .
            ?property schema:description ?propertyDescription .
            ?property skos:altLabel ?propertyAltLabel .
          }}
        }}
        """
        
        try:
            # Apply rate limiting before making SPARQL request
            rate_limiter.wait_if_needed()
            
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(timeout)
            
            query_results = sparql.query()
            sparql_results = query_results.convert()
            bindings: Dict = sparql_results["results"]["bindings"] # type: ignore
            
            # Create a mapping from PID to result data
            property_data = {}
            for result in bindings:
                property_uri = result.get("property", {}).get("value", "")
                pid = property_uri.split("/")[-1] if property_uri else ""
                
                if pid not in property_data:
                    property_data[pid] = {
                        'Property': pid,
                        'Title': result.get("propertyLabel", {}).get("value", ""),
                        'Description': result.get("propertyDescription", {}).get("value", ""),
                        'Alias': '',
                    }
                
                # Handle multiple aliases
                alt_label = result.get("propertyAltLabel", {}).get("value", "")
                if alt_label and alt_label not in property_data[pid]['Alias']:
                    if property_data[pid]['Alias']:
                        property_data[pid]['Alias'] += "|" + alt_label
                    else:
                        property_data[pid]['Alias'] = alt_label
            
            # Add results for all PIDs in this batch (including those not found)
            for pid in valid_pids:
                if pid in property_data:
                    results.append(property_data[pid])
                else:
                    # Property not found, add empty result
                    results.append(empty_results)
                    
        except Exception as e:
            print(f"Error in batch SPARQL query for properties: {e}")
            # Add empty results for this batch
            for pid in valid_pids:
                results.append(empty_results)
    
    return results


def fetch_relationship_triplet(prop: str, limit: int = None) -> Tuple[Set[Tuple[str, str, str]], Dict[str, str]]:
    """
    Fetches triplet relationships for a property from Wikidata.

    Args:
        prop (str): Property identifier of the relationship.

    Returns:
        List[List[str]]: A list of triplets (head, relation, tail) related to the property.
    """
    assert prop and 'P' == prop[0], "Your Property ID must be prefixed by a P"

    if limit is None: limit = math.inf

    client = get_thread_local_client()
    rel = client.get(EntityId(prop), load=True)

    triplets: set[tuple[str, str, str]] = set()
    rel_data = rel.data

    forward_dict = {}
    if rel.id != prop: forward_dict[rel.id] = prop

    if rel_data is None:
        raise ValueError(f"Property {prop} not found in Wikidata.")
    rel_claims = rel_data["claims"]
    if not isinstance(rel_claims, dict):
        raise ValueError(f"Property {prop} is not the expected type (dict).")

    for relation in rel_claims.keys():
        for statement in rel_claims[relation]:
            if len(triplets) > limit: break
            if ('datavalue' in statement['mainsnak'].keys()
                and isinstance(statement['mainsnak']['datavalue']['value'], dict)
                and 'id' in statement['mainsnak']['datavalue']['value'].keys()
                and 'P' == statement['mainsnak']['datavalue']['value']['id'][0]):

                triplet = (prop, relation, statement['mainsnak']['datavalue']['value']['id'])

                if not all([isinstance(elem, str) for elem in triplet]):
                    raise ValueError(f"An Element in triplet {triplet} is not a string")

                triplets.add(triplet)

        if len(triplets) > limit: break

    return triplets, forward_dict


def fetch_properties_sublist(offset: int, limit: int) -> List[str]:
    """
    Return a slice of property IDs (e.g., ['P10','P22',...]) using WDQS.
    This avoids scraping Special:ListProperties and complies with Wikimedia UA policy.
    """
    # Guard the inputs
    limit = max(1, int(limit))
    offset = max(0, int(offset))

    query = f"""
    SELECT ?p WHERE {{
      ?p a wikibase:Property .
    }}
    ORDER BY ?p
    LIMIT {limit}
    OFFSET {offset}
    """

    try:
        spq = get_sparql()          # uses your compliant UA + maxlag
        spq.setQuery(query)
        spq.setMethod("POST")
        spq.setTimeout(60)
        results = spq.query().convert()

        props: List[str] = []
        for row in results.get("results", {}).get("bindings", []):
            uri = row.get("p", {}).get("value", "")
            if uri:
                pid = uri.rsplit("/", 1)[-1]  # 'P123' from '.../entity/P123'
                if pid.startswith("P"):
                    props.append(pid)
        return props
    except Exception:
        return []


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

