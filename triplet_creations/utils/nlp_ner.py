# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:53:47 2024

@author: Eduin Hernandez

Summary: This script provides utility functions for named entity extraction and disambiguation, 
     mapping extracted entities to the closest match in Wikidata. It uses BERT-based sentence embeddings 
     and FAISS for approximate nearest neighbor search, ensuring that entities are matched accurately 
     using embedding similarity.

"""

import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from utils.wikidata_v2 import search_wikidata_relevant_id

from typing import List, Tuple, Dict

def capitalize(text: str) -> str:
    """Capitalizes the first letter of each word in the input text."""
    return ' '.join([word.capitalize() for word in text.split()])

def contains_stopwords(text: str, stopwords: set) -> Tuple[bool, set]:
    """Checks if the input text contains stopwords and returns a tuple of a boolean and the set of found stopwords."""
    words = text.split()
    found_stopwords = [word for word in words if word.lower() in stopwords]
    return len(found_stopwords) > 0, set(found_stopwords)

def extract_entities(doc: dict, stop_words: set):
    """Extracts entities from a given document, splitting them based on various criteria."""
    return [entity for ent in doc for entity in split_entities(ent['word'], stop_words)]

def split_entities(text: str, stopwords: set) -> List[str]:
    """Splits entity strings into components based on delimiters such as '&', 'and', and commas."""
    entities = []
    has_stopwords, useless_words = contains_stopwords(text, stopwords)
    if (' & ' in text or ' and ' in text) and ', ' in text:
        sub_entities = text.split(', ')
        for sub_ent in sub_entities:
            if ' & ' in sub_ent:
                entities.append([sub_ent, sub_ent.split(' & ')])
            elif ' and ' in sub_ent:
                entities.append([sub_ent, sub_ent.split(' and ')])
            else:
                entities.append(sub_ent)
    elif ' & ' in text:
        entities.append([text, text.split(' & ')])
    elif '&' in text:
        entities.append([text, text.replace('&', '')])
    elif ' and ' in text:
        entities.append([text, text.split(' and ')])
    elif ', ' in text:
        entities.extend(text.split(', '))
    elif has_stopwords:
        text_sub = [ent for ent in text.split() if ent not in useless_words]
        entities.append([text, [' '.join(text_sub)]])
    else:
        entities.append(text)
    return entities

def ann_judge(embeddings: np.ndarray, target_embedding: List[float]) -> Tuple[float, int]:
    """Performs approximate nearest neighbor (ANN) search to find the closest embedding using FAISS."""
    ann = faiss.IndexFlatL2(embeddings.shape[1])
    ann.add(embeddings)
    dist, index = ann.search(np.array(target_embedding)[None,:], 1)
    return dist[0,0], index[0,0]

def remove_duplicate_entities(entities_list: List[Dict]) -> List[Dict]:
    """Removes duplicate entities from the input list based on their QIDs."""
    seen_qids = set()
    return [entity for entity in entities_list if entity['Qid'] not in seen_qids and not seen_qids.add(entity['Qid'])]

def guess_wiki_entity(token: str, target_embedding: List[float], embedder: SentenceTransformer,
                      topk: int = 3, d_thres: float = 1.60) -> Dict:
    """Guesses the closest Wikidata entity for a given token using embedding distance for disambiguation."""
    # Handle token input
    if isinstance(token, str):
        guesses = search_wikidata_relevant_id(token, topk=topk)
        
        # Return early if no guesses are found
        if not guesses: return {}
    
        # Encode guesses and perform approximate nearest neighbor search
        embeddings = np.array([embedder.encode(f"{t0['Title']} {t0['Description']}") for t0 in guesses])
        dist, index = ann_judge(embeddings, target_embedding)
    
        # Return the best guess if the distance is within the threshold
        if dist <= d_thres: return guesses[index]
            # print(f"** {guesses[index]['Title']} - {dist}")
            
    # Handle token as a list with full and partial components
    elif isinstance(token, list) and len(token) == 2:
        full, partial = token
        f_guess, f_dist = guess_full_entity(full, target_embedding, embedder, topk, d_thres)
        p_guess, p_dist = guess_partial_entities(partial, target_embedding, embedder, topk, d_thres)

        if f_dist < p_dist: return f_guess
            # print(f"** {f_guess['Title']} - {f_dist}")
        elif p_guess: return p_guess
            # print(f"** {[a0['Title'] for a0 in p_guess]} - {p_dist}")
    return {}

# Helper functions for improved entity guessing
def guess_full_entity(full: str, target_embedding: List[float], embedder: SentenceTransformer, topk: int, d_thres: float) -> Tuple[Dict, float]:
    """Helper function to guess a full entity's closest match in Wikidata."""
    guesses = search_wikidata_relevant_id(full, topk=topk)
    if guesses:
        embeddings = np.array([embedder.encode(f"{t0['Title']} {t0['Description']}") for t0 in guesses])
        dist, index = ann_judge(embeddings, target_embedding)
        if dist <= d_thres:
            return guesses[index], dist
    return {}, float('inf')

def guess_partial_entities(partial: List[str], target_embedding: List[float], embedder: SentenceTransformer, topk: int, d_thres: float) -> Tuple[List[Dict], float]:
    """Helper function to guess the closest matches for partial components of an entity."""
    p_guess = []
    p_dist_total = 0
    for p in partial:
        p_guesses = search_wikidata_relevant_id(p, topk=topk)
        if p_guesses:
            p_embeddings = np.array([embedder.encode(f"{t0['Title']} {t0['Description']}") for t0 in p_guesses])
            dist, index = ann_judge(p_embeddings, target_embedding)
            if dist <= d_thres:
                p_guess.append(p_guesses[index])
                p_dist_total += dist

    if p_guess:
        avg_dist = p_dist_total / len(p_guess)
        return p_guess, avg_dist

    return [], float('inf')